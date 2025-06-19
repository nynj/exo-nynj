import argparse
import asyncio
import atexit
import signal
import json
import platform
import os
import time
import traceback
import uuid
from pathlib import Path
import numpy as np
from tqdm import tqdm
from exo.train.dataset import load_dataset, iterate_batches
from exo.networking.manual.manual_discovery import ManualDiscovery
from exo.orchestration.node import Node
from exo.networking.grpc.grpc_server import GRPCServer
from exo.networking.udp.udp_discovery import UDPDiscovery
from exo.networking.tailscale.tailscale_discovery import TailscaleDiscovery
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.api import ChatGPTAPI
from exo.download.shard_download import ShardDownloader, NoopShardDownloader
from exo.download.download_progress import RepoProgressEvent
from exo.download.new_shard_download import new_shard_downloader, has_exo_home_read_access, has_exo_home_write_access, ensure_exo_home, seed_models
from exo.helpers import print_yellow_exo, find_available_port, DEBUG, get_system_info, get_or_create_node_id, get_all_ip_addresses_and_interfaces, terminal_link, shutdown, get_device_capabilities_json
from exo.inference.shard import Shard
from exo.inference.inference_engine import get_inference_engine
from exo.inference.tokenizers import resolve_tokenizer
from exo.models import build_base_shard, get_repo, load_additional_models
from exo.viz.topology_viz import TopologyViz
import uvloop
import concurrent.futures
import resource
import psutil

# TODO: figure out why this is happening
os.environ["GRPC_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Configure uvloop for maximum performance
def configure_uvloop():
    uvloop.install()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Increase file descriptor limits on Unix systems
    if not psutil.WINDOWS:
      soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
      try: resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
      except ValueError:
        try: resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))
        except ValueError: pass

    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) * 4)))
    return loop

# parse args
parser = argparse.ArgumentParser(description="Initialize GRPC Discovery")
parser.add_argument("command", nargs="?", choices=["run", "eval", "train"], help="Command to run")
parser.add_argument("model_name", nargs="?", help="Model name to run")
parser.add_argument("--default-model", type=str, default=None, help="Default model")
parser.add_argument("--iters", type=int, default=100, help="Training iterations")
parser.add_argument("--save-every", type=int, default=5, help="Save the model every N iterations.")
parser.add_argument("--data", type=str, default="exo/train/data/lora", help="Directory where training data lives")
parser.add_argument("--batch-size", type=int, default=1, help="Minibatch size.")
parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to a custom checkpoint to load")
parser.add_argument("--save-checkpoint-dir", type=str, default="checkpoints", help="Path to a folder where checkpoints are stored")
parser.add_argument("--node-id", type=str, default=None, help="Node ID")
parser.add_argument("--node-host", type=str, default="0.0.0.0", help="Node host")
parser.add_argument("--node-port", type=int, default=None, help="Node port")
parser.add_argument("--models-seed-dir", type=str, default=None, help="Model seed directory")
parser.add_argument("--listen-port", type=int, default=5678, help="Listening port for discovery")
parser.add_argument("--download-quick-check", action="store_true", help="Quick check local path for model shards download")
parser.add_argument("--max-parallel-downloads", type=int, default=8, help="Max parallel downloads for model shards download")
parser.add_argument("--broadcast-port", type=int, default=5678, help="Broadcast port for discovery")
parser.add_argument("--discovery-module", type=str, choices=["udp", "tailscale", "manual"], default="udp", help="Discovery module to use")
parser.add_argument("--discovery-timeout", type=int, default=30, help="Discovery timeout in seconds")
parser.add_argument("--discovery-config-path", type=str, default=None, help="Path to discovery config json file")
parser.add_argument("--get-device-capabilities", action="store_true", help="Output the current device's auto-detected capabilities in JSON format and exit")
parser.add_argument("--wait-for-peers", type=int, default=0, help="Number of peers to wait to connect to before starting")
parser.add_argument("--chatgpt-api-port", type=int, default=52415, help="ChatGPT API port")
parser.add_argument("--chatgpt-api-response-timeout", type=int, default=900, help="ChatGPT API response timeout in seconds")
parser.add_argument("--max-generate-tokens", type=int, default=10000, help="Max tokens to generate in each request")
parser.add_argument("--inference-engine", type=str, default=None, help="Inference engine to use (mlx, tinygrad, or dummy)")
parser.add_argument("--disable-tui", action=argparse.BooleanOptionalAction, help="Disable TUI")
parser.add_argument("--run-model", type=str, help="Specify a model to run directly")
parser.add_argument("--prompt", type=str, help="Prompt for the model when using --run-model", default="Who are you?")
parser.add_argument("--default-temp", type=float, help="Default token sampling temperature", default=0.0)
parser.add_argument("--tailscale-api-key", type=str, default=None, help="Tailscale API key")
parser.add_argument("--tailnet-name", type=str, default=None, help="Tailnet name")
parser.add_argument("--node-id-filter", type=str, default=None, help="Comma separated list of allowed node IDs (only for UDP and Tailscale discovery)")
parser.add_argument("--interface-type-filter", type=str, default=None, help="Comma separated list of allowed interface types (only for UDP discovery)")
parser.add_argument("--system-prompt", type=str, default=None, help="System prompt for the ChatGPT API")
parser.add_argument("--additional-models", type=str, default=None, help="A JSON file of additional models to serve")
args = parser.parse_args()

# Handle the --get-device-capabilities option before printing anything else so it can be used for automation
if args.get_device_capabilities:
    print(get_device_capabilities_json())
    exit(0)

print(f"Selected inference engine: {args.inference_engine}")

print_yellow_exo()

system_info = get_system_info()
print(f"Detected system: {system_info}")

shard_downloader: ShardDownloader = new_shard_downloader(args.max_parallel_downloads) if args.inference_engine != "dummy" else NoopShardDownloader()
inference_engine_name = args.inference_engine or ("mlx" if system_info == "Apple Silicon Mac" else "tinygrad")
print(f"Inference engine name after selection: {inference_engine_name}")

inference_engine = get_inference_engine(inference_engine_name, shard_downloader)
print(f"Using inference engine: {inference_engine.__class__.__name__} with shard downloader: {shard_downloader.__class__.__name__}")

if args.node_port is None:
  args.node_port = find_available_port(args.node_host)
  if DEBUG >= 1: print(f"Using available port: {args.node_port}")

args.node_id = args.node_id or get_or_create_node_id()
chatgpt_api_endpoints = [f"http://{ip}:{args.chatgpt_api_port}/v1/chat/completions" for ip, _ in get_all_ip_addresses_and_interfaces()]
web_chat_urls = [f"http://{ip}:{args.chatgpt_api_port}" for ip, _ in get_all_ip_addresses_and_interfaces()]
if DEBUG >= 0:
  print("Chat interface started:")
  for web_chat_url in web_chat_urls:
    print(f" - {terminal_link(web_chat_url)}")
  print("ChatGPT API endpoint served at:")
  for chatgpt_api_endpoint in chatgpt_api_endpoints:
    print(f" - {terminal_link(chatgpt_api_endpoint)}")

# Convert node-id-filter and interface-type-filter to lists if provided
allowed_node_ids = args.node_id_filter.split(',') if args.node_id_filter else None
allowed_interface_types = args.interface_type_filter.split(',') if args.interface_type_filter else None

if args.discovery_module == "udp":
  discovery = UDPDiscovery(
    args.node_id,
    args.node_port,
    args.listen_port,
    args.broadcast_port,
    lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
    discovery_timeout=args.discovery_timeout,
    allowed_node_ids=allowed_node_ids,
    allowed_interface_types=allowed_interface_types
  )
elif args.discovery_module == "tailscale":
  discovery = TailscaleDiscovery(
    args.node_id,
    args.node_port,
    lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
    discovery_timeout=args.discovery_timeout,
    tailscale_api_key=args.tailscale_api_key,
    tailnet=args.tailnet_name,
    allowed_node_ids=allowed_node_ids
  )
elif args.discovery_module == "manual":
  # Manual discovery reads peer information from a JSON configuration file.
  if not args.discovery_config_path:
    # This mode requires a configuration file path.
    raise ValueError(f"--discovery-config-path is required when using manual discovery. Please provide a path to a config json file.")
  # Manual discovery uses a JSON config file that defines all nodes in the network
  # The config file should contain a "peers" object mapping node_ids to their connection details
  # See NetworkTopology class in exo/networking/manual/network_topology_config.py for the expected format
  discovery = ManualDiscovery(args.discovery_config_path, args.node_id, create_peer_handle=lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities))
# Initialize the TopologyViz TUI if it's not disabled.
topology_viz = TopologyViz(chatgpt_api_endpoints=chatgpt_api_endpoints, web_chat_urls=web_chat_urls) if not args.disable_tui else None

# Load additional models from a JSON file if specified.
if args.additional_models is not None:
  path = Path(args.additional_models)
  # Ensure the file exists before attempting to load it.
  if not path.exists():
    raise ValueError(f"Additional models file {path} does not exist")

  # Load the models into the model registry.
  load_additional_models(path)

# --- Core Application Objects Initialization ---
# Create the main Node object. This is the heart of the application logic.
node = Node(
  args.node_id,
  None, # The shard is managed by the node itself.
  inference_engine,
  discovery,
  shard_downloader,
  partitioning_strategy=RingMemoryWeightedPartitioningStrategy(),
  max_generate_tokens=args.max_generate_tokens,
  topology_viz=topology_viz,
  default_sample_temperature=args.default_temp
)
# Create the gRPC server to handle communication between nodes.
server = GRPCServer(node, args.node_host, args.node_port)
# Link the server to the node.
node.server = server
# Create the ChatGPT-compatible API server.
api = ChatGPTAPI(
  node,
  node.inference_engine.__class__.__name__,
  response_timeout=args.chatgpt_api_response_timeout,
  on_chat_completion_request=lambda req_id, __, prompt: topology_viz.update_prompt(req_id, prompt) if topology_viz else None,
  default_model=args.default_model,
  system_prompt=args.system_prompt
)
# --- Event Handlers ---
# A buffer to store token outputs for each request before updating the TUI.
buffered_token_output = {}
# This function updates the TUI with the generated tokens for a request.
def update_topology_viz(req_id, tokens, __, ___):
  # Do nothing if the TUI is disabled.
  if not topology_viz: return
  # Do nothing if there's no active shard.
  if not node.inference_engine.shard: return
  # Ignore token updates for image generation models.
  if node.inference_engine.shard.model_id == 'stable-diffusion-2-1-base': return
  # Append new tokens to the buffer for the given request ID.
  if req_id in buffered_token_output: buffered_token_output[req_id].extend(tokens)
  else: buffered_token_output[req_id] = tokens
  # Decode the tokens and update the prompt output in the TUI.
  topology_viz.update_prompt_output(req_id, node.inference_engine.tokenizer.decode(buffered_token_output[req_id]))
# Register the function to be called on each token generation event.
node.on_token.register("update_topology_viz").on_next(update_topology_viz)
# This function updates the TUI with the initial prompt when a request starts processing.
def update_prompt_viz(request_id, opaque_status: str):
  # Do nothing if the TUI is disabled.
  if not topology_viz: return
  try:
    # The status is passed as a JSON string.
    status = json.loads(opaque_status)
    # Check if this status update is for the start of prompt processing.
    if status.get("type") != "node_status" or status.get("status") != "start_process_prompt": return
    # Update the TUI with the prompt.
    topology_viz.update_prompt(request_id, status.get("prompt", "corrupted prompt (this should never happen)"))
  except Exception as e:
    # Log any errors that occur during the update.
    if DEBUG >= 2:
      print(f"Failed to update prompt viz: {e}")
      traceback.print_exc()
# Register the function to be called on opaque status update events.
node.on_opaque_status.register("update_prompt_viz").on_next(update_prompt_viz)

# This function preemptively starts downloading a model shard when a request is received for it.
def preemptively_load_shard(request_id: str, opaque_status: str):
  try:
    # Parse the status JSON.
    status = json.loads(opaque_status)
    # Check if the status indicates the start of prompt processing.
    if status.get("type") != "node_status" or status.get("status") != "start_process_prompt": return
    # Get the shard information from the status.
    current_shard = node.get_current_shard(Shard.from_dict(status.get("shard")))
    if DEBUG >= 2: print(f"Preemptively starting download for {current_shard}")
    # Start the download as a background task.
    asyncio.create_task(node.inference_engine.ensure_shard(current_shard))
  except Exception as e:
    # Log any errors.
    if DEBUG >= 2:
      print(f"Failed to preemptively start download: {e}")
      traceback.print_exc()
# Register the function to be called on opaque status update events.
node.on_opaque_status.register("preemptively_load_shard").on_next(preemptively_load_shard)

# This dictionary stores the last broadcasted progress event for each shard.
last_events: dict[str, tuple[float, RepoProgressEvent]] = {}
# This function broadcasts download progress to other nodes, with throttling.
def throttled_broadcast(shard: Shard, event: RepoProgressEvent):
  global last_events
  current_time = time.time()
  # Don't broadcast 'not_started' events.
  if event.status == "not_started": return
  last_event = last_events.get(shard.model_id)
  # Avoid sending duplicate 'complete' events.
  if last_event and last_event[1].status == "complete" and event.status == "complete": return
  # Throttle updates to one every 0.2 seconds for the same status.
  if last_event and last_event[0] == event.status and current_time - last_event[0] < 0.2: return
  # Store the current event and time.
  last_events[shard.model_id] = (current_time, event)
  # Broadcast the progress update as an opaque status.
  asyncio.create_task(node.broadcast_opaque_status("", json.dumps({"type": "download_progress", "node_id": node.id, "progress": event.to_dict()})))
# Register the function to be called on shard downloader progress events.
shard_downloader.on_progress.register("broadcast").on_next(throttled_broadcast)

# --- CLI Command Implementations ---

# This function handles the 'run' command to process a single prompt from the CLI.
async def run_model_cli(node: Node, model_name: str, prompt: str):
  # Get the class name of the current inference engine.
  inference_class = node.inference_engine.__class__.__name__
  # Build the base shard for the requested model.
  shard = build_base_shard(model_name, inference_class)
  # If the model is not supported, print an error and return.
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
    return
  # Resolve the tokenizer for the model.
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  # Generate a unique ID for this request.
  request_id = str(uuid.uuid4())
  # Create a callback ID to wait for the response.
  callback_id = f"cli-wait-response-{request_id}"
  # Register a one-time callback to receive the generated tokens.
  callback = node.on_token.register(callback_id)
  # Update the TUI with the prompt, if enabled.
  if topology_viz:
    topology_viz.update_prompt(request_id, prompt)
  # Apply the chat template to the prompt.
  prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

  try:
    print(f"Processing prompt: {prompt}")
    # Send the prompt processing request to the node.
    await node.process_prompt(shard, prompt, request_id=request_id)

    tokens = []
    # This function is called for each token event and collects the tokens.
    def on_token(_request_id, _tokens, _is_finished):
      tokens.extend(_tokens)
      # It returns True when the response for our request is complete.
      return _request_id == request_id and _is_finished
    # Wait for the response to be completed.
    await callback.wait(on_token, timeout=300)

    # Print the decoded response.
    print("\nGenerated response:")
    print(tokenizer.decode(tokens))
  except Exception as e:
    # Print any errors that occur.
    print(f"Error processing prompt: {str(e)}")
    traceback.print_exc()
  finally:
    # Clean up by deregistering the callback.
    node.on_token.deregister(callback_id)

# Utility function to clean and expand user path.
def clean_path(path):
    """Clean and resolve path"""
    # Handles paths that might be wrapped in "Optional(...)"
    if path.startswith("Optional("):
        path = path.strip('Optional("').rstrip('")')
    # Expands the '~' to the user's home directory.
    return os.path.expanduser(path)

# This function waits until all outstanding requests on the node are finished.
async def hold_outstanding(node: Node):
  # It checks every half second.
  while node.outstanding_requests:
    await asyncio.sleep(.5)
  return

# This function runs one iteration of training or evaluation.
async def run_iter(node: Node, shard: Shard, train: bool, data, batch_size=1):
  losses = []
  tokens = []
  # Iterate over batches of data.
  for batch in tqdm(iterate_batches(data, batch_size), total=len(data) // batch_size):
    _, _, lengths = batch
    # Enqueue the example for processing and calculate the loss.
    losses.append(np.sum(lengths * await node.enqueue_example(shard, *batch, train=train)))
    # Keep track of the number of tokens processed.
    tokens.append(np.sum(lengths))
  # Calculate the total number of tokens and the average loss.
  total_tokens = np.sum(tokens)
  total_loss = np.sum(losses) / total_tokens

  return total_loss, total_tokens

# This function handles the 'eval' command to evaluate a model.
async def eval_model_cli(node: Node, model_name, dataloader, batch_size, num_batches=-1):
  # Get the inference class name.
  inference_class = node.inference_engine.__class__.__name__
  # Build the shard for the model.
  shard = build_base_shard(model_name, inference_class)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
    return
  # Load the tokenizer and dataset.
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  train, val, test = dataloader(tokenizer.encode)
  print(f"Evaluating {len(test)} examples with batch_size {batch_size}")
  # Run the evaluation iteration.
  loss, tokens = await run_iter(node, shard, False, test, batch_size)
  # Print the results.
  print(f"total | {loss=}, {tokens=}")
  print("Waiting for outstanding tasks")
  # Wait for all tasks to complete before exiting.
  await hold_outstanding(node)

# This function handles the 'train' command to train a model.
async def train_model_cli(node: Node, model_name, dataloader, batch_size, iters, save_interval=0, checkpoint_dir=None):
  # Get the inference class name.
  inference_class = node.inference_engine.__class__.__name__
  # Build the shard for the model.
  shard = build_base_shard(model_name, inference_class)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
    return
  # Load the tokenizer and dataset.
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  train, val, test = dataloader(tokenizer.encode)
  print(f"Training on {len(train)} examples with batch_size {batch_size} for {iters} epochs")
  # A short delay to allow the network to stabilize.
  for i in tqdm(range(3)):
    await asyncio.sleep(1)
  # Loop for the specified number of training epochs.
  for epoch in range(iters):
    # Run a training iteration.
    loss, tokens = await run_iter(node, shard, True, train, batch_size)
    print(f"epoch {epoch + 1}/{iters}\t| loss: {loss}, tokens: {tokens}")
    # Save a checkpoint if the save interval is met.
    if save_interval > 0 and epoch > 0 and (epoch % save_interval) == 0 and checkpoint_dir is not None:
      # Coordinate saving the model state across the network.
      await node.coordinate_save(shard, epoch, checkpoint_dir)
      # Wait for the save operation to complete.
      await hold_outstanding(node)
  # Wait for all outstanding tasks to complete before exiting.
  await hold_outstanding(node)

# This function checks for the existence and permissions of the exo home directory.
async def check_exo_home():
  # Get the path and permissions for the exo home directory.
  home, has_read, has_write = await ensure_exo_home(), await has_exo_home_read_access(), await has_exo_home_write_access()
  if DEBUG >= 1: print(f"exo home directory: {home}")
  print(f"{has_read=}, {has_write=}")
  # If permissions are insufficient, print a warning.
  if not has_read or not has_write:
    print(f"""
          WARNING: Limited permissions for exo home directory: {home}.
          This may prevent model downloads from working correctly.
          {"❌ No read access" if not has_read else ""}
          {"❌ No write access" if not has_write else ""}
          """)

# --- Main Application Logic ---
async def main():
  # Get the current asyncio event loop.
  loop = asyncio.get_running_loop()

  # Check the exo home directory.
  try: await check_exo_home()
  except Exception as e: print(f"Error checking exo home directory: {e}")

  # Seed models from a local directory if specified.
  if not args.models_seed_dir is None:
    try:
      models_seed_dir = clean_path(args.models_seed_dir)
      # This copies model files to the exo home directory.
      await seed_models(models_seed_dir)
    except Exception as e:
      print(f"Error seeding models: {e}")

  # This function restores the terminal cursor on exit.
  def restore_cursor():
    if platform.system() != "Windows":
        # 'tput cnorm' is the command to make the cursor visible.
        os.system("tput cnorm")

  # Register the restore_cursor function to be called at program exit.
  atexit.register(restore_cursor)

  # This function handles graceful shutdown on receiving an exit signal.
  def handle_exit():
    # Schedule the shutdown coroutine to run on the event loop.
    asyncio.ensure_future(shutdown(signal.SIGTERM, loop, node.server))

  # Register signal handlers for SIGINT (Ctrl+C) and SIGTERM on non-Windows systems.
  if platform.system() != "Windows":
    for s in [signal.SIGINT, signal.SIGTERM]:
      loop.add_signal_handler(s, handle_exit)

  # Start the node, which includes starting discovery and connecting to peers.
  await node.start(wait_for_peers=args.wait_for_peers)

  # --- Command Dispatch ---
  # Execute the specified command.
  if args.command == "run" or args.run_model:
    # Determine the model name from either 'run' command or '--run-model' argument.
    model_name = args.model_name or args.run_model
    if not model_name:
      print("Error: Model name is required when using 'run' command or --run-model")
      return
    # Run a single inference.
    await run_model_cli(node, model_name, args.prompt)
  elif args.command == "eval" or args.command == 'train':
    model_name = args.model_name
    # Create a dataloader for the training/evaluation data.
    dataloader = lambda tok: load_dataset(args.data, preprocess=lambda item: tok(item)
                                                   , loadline=lambda line: json.loads(line).get("text",""))
    if args.command == 'eval':
      if not model_name:
        print("Error: Much like a human, I can't evaluate anything without a model")
        return
      # Evaluate the model.
      await eval_model_cli(node, model_name, dataloader, args.batch_size)
    else: # 'train'
      if not model_name:
        print("Error: This train ain't leaving the station without a model")
        return
      # Train the model.
      await train_model_cli(node, model_name, dataloader, args.batch_size, args.iters, save_interval=args.save_every, checkpoint_dir=args.save_checkpoint_dir)

  else:
    # If no command is specified, run in persistent server mode.
    # Start the API server as a background task.
    asyncio.create_task(api.run(port=args.chatgpt_api_port))
    # Wait indefinitely until an exit signal is received.
    await asyncio.Event().wait()

  # After a command finishes or the server is shut down, give peers time to disconnect gracefully.
  if args.wait_for_peers > 0:
    print("Cooldown to allow peers to exit gracefully")
    # This loop provides a visual progress bar for the cooldown period.
    for i in tqdm(range(50)):
      await asyncio.sleep(.1)

# This function sets up the uvloop and runs the main async function.
def run():
    loop = None
    try:
        # Configure and get the high-performance uvloop.
        loop = configure_uvloop()
        # Run the main function until it completes.
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        # Handle Ctrl+C to gracefully shut down.
        print("\nShutdown requested... exiting")
    finally:
        # Close the event loop on exit.
        if loop: loop.close()

if __name__ == "__main__":
  run()
