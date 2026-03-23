# Superior Agent 🧠

Superior Agent is a modular, session-aware AI agent designed with tiered reasoning, powered by **Qwen3** via **Ollama**. This is a fun hobbyist project exploring the capabilities of autonomous agents in a controlled environment.

## ✨ Key Features

- **Tiered Reasoning**: Efficiently processes tasks using a structured thinking-to-action pipeline.
- **Dynamic Tool Discovery**: Automatically discovers tools in the `agent/tools/` directory using AST parsing—no manual registration required!
- **Lazy-Loading Tools**: Tools are only imported and loaded when the agent actually needs them, keeping the startup light.
- **Session Awareness**: Tracks artifacts and memory across the current session.
- **Rich Terminal UI**: Beautifully formatted output using `rich` and `textual`.

## 🛠 Tools System: Adding Your Own

Implementing custom tools is incredibly easy. Each tool is a standalone `.py` file in `superior_agent/agent/tools/`.

### How to Create a Tool

1.  Create a new Python file (e.g., `my_tool.py`).
2.  Define **exactly one** public function.
3.  Add a specifically formatted docstring. The agent uses this to understand how and when to use your tool.

**Example (`get_current_time.py`):**

```python
import datetime

def get_current_time() -> str:
    """Description: Retrieves the current local date and time.
    Returns: The current date and time as a string.
    When to use: When the user asks for the current time or date.
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
```

The `Registry` will automatically pick it up, parse the `Description`, `Args`, `Returns`, and `When to use` fields, and make it available to the agent.

## 🚀 Getting Started

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/SuperiorAgent.git
    cd SuperiorAgent
    ```

2.  **Install dependencies**:
    We recommend using a virtual environment.
    ```bash
    pip install -e .
    ```
    For development dependencies (testing):
    ```bash
    pip install -e ".[dev]"
    ```

### Running the Agent

You can run the agent as a module:

```bash
python -m superior_agent
```

#### Running Locally (Ollama)
Ensure you have [Ollama](https://ollama.ai/) installed and running. By default, the agent looks for Ollama at `http://localhost:11434` and uses the `qwen3:latest` model.

#### Running via Cloud
If you want to use a cloud-hosted Ollama-compatible API:
```bash
python -m superior_agent --ollama-url https://your-cloud-api.com --apikey YOUR_API_KEY
```

## ⚙️ Command-Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--model` | Preferred Ollama model tag | `qwen3:latest` |
| `--ollama-url` | Base URL for the Ollama API | `http://localhost:11434` |
| `--apikey` | API Key for the request (x-api-key header) | `None` |
| `--workdir` | Base directory for file-based tools | `cwd` |
| `--debug` | Enable detailed debug logging | `False` |

## 🧪 Testing

The project uses `pytest` for automated testing. To run the suite:

```bash
pytest
```

---
*Happy Agentic Coding!* 🤖✨
