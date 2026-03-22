from dataclasses import dataclass
from typing import Set

@dataclass
class AgentTemplate:
    name: str
    description: str
    system_prompt_prefix: str
    initial_tools: Set[str]

TEMPLATES = {
    "General": AgentTemplate(
        name="General",
        description="A balanced assistant for general tasks.",
        system_prompt_prefix="You are Superior Agent, a versatile autonomous assistant. You handle a wide range of tasks with a balanced approach to logic, creativity, and system operations.",
        initial_tools={"search_tools", "get_session_info", "update_artifact", "increase_max_rounds"}
    ),
    "Coding": AgentTemplate(
        name="Coding",
        description="Expert software engineer focused on code and system tasks.",
        system_prompt_prefix="You are Superior Agent, an expert software engineer. Your focus is on writing clean, efficient, and well-documented code. You are proficient in system operations, debugging, and architectural design.",
        initial_tools={
            "read_file", "write_file", "edit_file", "run_shell", 
            "list_directory", "list_processes", "stop_process", 
            "search_tools", "get_session_info", "update_artifact", "increase_max_rounds"
        }
    ),
    "Research": AgentTemplate(
        name="Research",
        description="Meticulous researcher optimized for information gathering.",
        system_prompt_prefix="You are Superior Agent, a meticulous researcher. Your goal is to gather accurate information, analyze complex data, and provide detailed, well-cited summaries. You prioritize depth and verification.",
        initial_tools={"search_tools", "get_session_info", "update_artifact", "increase_max_rounds"}
    ),
}
