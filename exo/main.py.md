# `main.py`: The Heart of the Exo Node

## Overview

`main.py` is the main entry point for the `exo` application. It's responsible for initializing and running an `exo` node, which can participate in a distributed network for running inference on large language models. The script can be used to start a persistent node that serves an API, or to run one-off tasks like model inference, training, or evaluation from the command line.

## Key Responsibilities

1.  **Argument Parsing**: Parses command-line arguments to configure the node's behavior, including which command to run (`run`, `eval`, `train`), model selection, network configuration, and more.
2.  **Node Initialization**: Sets up and configures the core `Node` object, which represents the current device in the `exo` network.
3.  **Component Wiring**: Initializes and connects various components:
    *   **Inference Engine**: Selects and initializes the appropriate backend for model inference (e.g., `MLX` for Apple Silicon, `TinyGrad`).
    *   **Networking**: Sets up the gRPC server for communication with other nodes.
    *   **Discovery**: Configures the mechanism for finding peer nodes (e.g., UDP broadcast, Tailscale, or a manual configuration).
    *   **Shard Downloader**: Manages downloading of model shards.
4.  **API Server**: Starts a ChatGPT-compatible API server to accept inference requests.
5.  **Command Execution**: Executes specific commands like `run` (for a single inference), `eval` (for model evaluation), and `train` (for model training).
6.  **TUI (Topology Visualization)**: Optionally, it can launch a Textual User Interface to visualize the state of the network and ongoing requests.

## Architecture

The script follows an asynchronous, event-driven architecture built on Python's `asyncio` and `uvloop` for high performance.

1.  **Configuration**: At startup, it reads command-line arguments to configure every aspect of the node.
2.  **Initialization**:
    *   It configures `uvloop` and resource limits.
    *   It determines the best `InferenceEngine` based on the system hardware.
    *   It creates a `Node` instance, passing it the inference engine, discovery mechanism, and other configurations.
    *   A `GRPCServer` is created and linked to the `Node`.
    *   A `ChatGPTAPI` wrapper is created around the `Node` to expose a standard interface.
3.  **Execution Flow**:
    *   If a command like `run`, `eval`, or `train` is provided, it executes the corresponding async function (`run_model_cli`, `eval_model_cli`, `train_model_cli`) and then exits.
    *   If no command is given, it starts the `ChatGPTAPI` server and enters a persistent state, waiting for API requests or instructions from other nodes. The `node.start()` method initiates discovery and peer connection.
4.  **Signal Handling**: It gracefully handles `SIGINT` and `SIGTERM` to shut down the node and its server.

## How to Use

You can run this script from the command line.

**To start a node and serve the API:**

```bash
python -m exo.main
```

This will start a node, which will discover other peers on the local network and be ready to accept inference requests.

**To run inference on a specific model:**

```bash
python -m exo.main run <model_name> --prompt "Your prompt here"
```

**To train a model:**

```bash
python -m exo.main train <model_name> --data <path_to_data>
```

Refer to the extensive list of command-line arguments within `main.py` for more advanced configuration options, such as setting up different discovery methods (`--discovery-module`), filtering nodes, or configuring the TUI. 