"""Entry point for ``python -m superior_agent``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .core.llm_bridge import LLMBridge
from .core.context_manager import ContextManager
from .agent.brain import Brain, detect_platform
from .agent.registry import Registry
from .agent.artifact_controller import ArtifactController
from .frontend.cli import CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="superior_agent",
        description="Superior Agent — a modular AI agent with tiered reasoning.",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Working directory for file operations (default: cwd)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3:latest",
        help="Default Ollama model tag (default: qwen3:latest)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--apikey",
        type=str,
        default=None,
        help="API key for Ollama requests (x-api-key header)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Logging
    level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=level,
        filename="superior_agent.log",
        filemode="a",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Platform detection
    workdir = args.workdir or str(Path.cwd())
    profile = detect_platform(workdir)

    # Core layer
    llm = LLMBridge(
        host=args.ollama_url,
        model=args.model,
        api_key=args.apikey,
    )

    # Agent layer
    registry = Registry()
    tools_dir = Path(__file__).parent / "agent" / "tools"
    registry.discover(tools_dir)

    import uuid
    session_id = uuid.uuid4().hex[:12]
    artifact_ctrl = ArtifactController(session_id)

    brain = Brain(
        llm_bridge=llm,
        registry=registry,
        artifact_ctrl=artifact_ctrl,
        platform_profile=profile,
    )

    # Frontend
    cli = CLI(brain=brain, artifact_ctrl=artifact_ctrl)

    # Run
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        pass
    finally:
        artifact_ctrl.close()
        asyncio.run(llm.close())


if __name__ == "__main__":
    main()
