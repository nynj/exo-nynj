from dataclasses import dataclass, field

# standard MLX imports
import mlx.core as mx
import mlx.nn as nn

# third-party Qwen3 definitions (from mlx_lm)
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_moe import (
  ModelArgs as Qwen3BaseArgs,  # base model args
  Qwen3MoeDecoderLayer,       # single transformer decoder block
)

# local helpers
from exo.inference.shard import Shard
from .base import IdentityBlock


# -------------------------------------------------------------
# 1. Extended ModelArgs with dynamic shard information
# -------------------------------------------------------------
@dataclass
class ModelArgs(Qwen3BaseArgs):
  """Wrapper around upstream Qwen3 `ModelArgs` adding a `shard` field.

  This dataclass is *JSON-serialisable* and therefore can be
  reconstructed from the config dict produced by the inference engine.
  """

  # shard configuration used by the dynamic sharded inference engine
  shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

  # -----------------------------------------------------------
  # post-processing to convert nested dict -> Shard instance
  # -----------------------------------------------------------
  def __post_init__(self):
    # Users can pass either an actual `Shard` or its dict-ified form.
    if isinstance(self.shard, Shard):
      return
    if not isinstance(self.shard, dict):
      raise TypeError(
        f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead"
      )
    # convert dict to Shard object for convenience
    self.shard = Shard(**self.shard)


# -------------------------------------------------------------
# 2. Backbone model restricted to *this* shard only
# -------------------------------------------------------------
class Qwen3Model(nn.Module):
  """Qwen3 backbone with layers outside this shard turned into no-ops."""

  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args

    # basic metadata
    self.vocab_size = args.vocab_size
    self.num_hidden_layers = args.num_hidden_layers

    # embeddings – first shard always needs them; last shard as well if
    # word embeddings are tied with the LM head.
    if args.shard.is_first_layer() or (
      args.shard.is_last_layer() and args.tie_word_embeddings
    ):
      # token embedding table
      self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

    # transformer blocks – replace blocks outside our [start, end] range
    self.layers = []
    for idx in range(self.num_hidden_layers):
      if args.shard.start_layer <= idx <= args.shard.end_layer:
        # real compute block
        self.layers.append(Qwen3MoeDecoderLayer(args, idx))  # idx helpful for weight loading
      else:
        # identity to keep tensor shape intact without compute
        self.layers.append(IdentityBlock())

    # final RMSNorm only on last shard
    if args.shard.is_last_layer():
      self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

  # -------------------------------------------------------------------
  # forward pass
  # -------------------------------------------------------------------
  def __call__(
    self,
    inputs: mx.array,                       # (B, T)
    cache: list[KVCache] | None = None,     # per-layer KV cache
  ) -> mx.array:

    # 1. embed tokens (first shard only)
    if self.args.shard.is_first_layer():
      h = self.embed_tokens(inputs)
    else:
      h = inputs

    # 2. build causal mask if sequence length > 1 (single-token path is faster)
    mask = None
    T = h.shape[1]
    if T > 1:
      mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
      mask = mask.astype(h.dtype)

    # 3. normalise cache list length
    if cache is None:
      cache = [None] * len(self.layers)

    # 4. iterate over (layer, layer-cache)
    for layer, c in zip(self.layers, cache):
      h = layer(h, mask, c)

    # 5. final RMSNorm (last shard only)
    if self.args.shard.is_last_layer():
      h = self.norm(h)
    return h


# -------------------------------------------------------------
# 3. Full model wrapper inc. (optional) lm_head
# -------------------------------------------------------------
class Model(nn.Module):
  """Shard-aware Qwen3 MoE wrapper compatible with Exo's inference engine."""

  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    self.model_type = args.model_type

    # backbone (possibly partial)
    self.model = Qwen3Model(args)

    # lm_head lives on last shard *unless* embeddings are tied
    if args.shard.is_last_layer():
      if not args.tie_word_embeddings:
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

  # -------------------------------------------------------------------
  def __call__(
    self,
    inputs: mx.array,
    cache: list[KVCache] | None = None,
  ):
    """Forward pass that returns raw hidden states or logits depending on shard."""
    out = self.model(inputs, cache)

    # if lm_head present / required convert to logits
    if self.args.shard.is_last_layer():
      if self.args.tie_word_embeddings:
        out = self.model.embed_tokens.as_linear(out)
      else:
        out = self.lm_head(out)
    return out

  # -----------------------------------------------------------
  # 4. Weight sanitisation – retain only params required for shard
  # -----------------------------------------------------------
  def sanitize(self, weights: dict[str, mx.array]):
    """Filter & transform HF/MLX weights so that only current shard is kept.

    Also merges expert weights (from separate experts) into the stacked
    `switch_mlp.*` tensors expected by Exo's inference engine.
    """

    shard_state_dict: dict[str, mx.array] = {}

    # ------------ 4a. keep parameters belonging to this shard ------------
    for key, value in weights.items():
      # skip rotary emb inverse frequencies – they are constant and large
      if "self_attn.rotary_emb.inv_freq" in key:
        continue

      # transformer block weights
      if key.startswith("model.layers."):
        layer_num = int(key.split(".")[2])
        if self.args.shard.start_layer <= layer_num <= self.args.shard.end_layer:
          shard_state_dict[key] = value
      # embeddings / lm_head / norm
      elif self.args.shard.is_first_layer() and key.startswith("model.embed_tokens"):
        shard_state_dict[key] = value
      elif (
        self.args.shard.is_last_layer() and self.args.tie_word_embeddings and key.startswith("model.embed_tokens")
      ):
        shard_state_dict[key] = value
      elif (
        self.args.shard.is_last_layer() and not self.args.tie_word_embeddings and key.startswith("lm_head")
      ):
        shard_state_dict[key] = value
      elif self.args.shard.is_last_layer() and key.startswith("model.norm"):
        shard_state_dict[key] = value

    # remove standalone lm_head if embeddings are tied
    if self.args.tie_word_embeddings:
      shard_state_dict.pop("lm_head.weight", None)

    # ------------ 4b. merge per-expert tensors into switch-MLP ------------
    # determine how many experts are present in checkpoint (Qwen3 uses `num_experts`,
    # DeepSeek style configs use `n_routed_experts`).
    n_experts: int = getattr(
      self.args,
      "n_routed_experts",  # DeepSeek / some configs
      getattr(self.args, "num_experts", 0)  # Qwen3 configs
    )

    if n_experts == 0:
      raise AttributeError(
        "Could not determine number of experts from ModelArgs. Expected either `n_routed_experts` or `num_experts`."
      )

    for l in range(self.args.num_hidden_layers):
      prefix = f"model.layers.{l}"
      # gating, down, up projections
      for m in ["gate_proj", "down_proj", "up_proj"]:
        for k in ["weight", "scales", "biases"]:
          expert_key = f"{prefix}.mlp.experts.0.{m}.{k}"
          if expert_key in shard_state_dict:
            # gather tensors across experts and stack along *new* dimension
            to_join = [
              shard_state_dict.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
              for e in range(n_experts)
            ]
            shard_state_dict[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

    return shard_state_dict

  # convenience properties used by the inference engine -----------------
  @property
  def layers(self):
    return self.model.layers

  @property
  def head_dim(self):
    # Standard definition (same as Qwen2): hidden_size / num_attention_heads
    return self.args.hidden_size // self.args.num_attention_heads

  @property
  def n_kv_heads(self):
    return self.args.num_key_value_heads
