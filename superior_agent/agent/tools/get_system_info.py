import platform

def get_system_info() -> str:
    """Description: Retrieves the system information.
    Returns: A list of strings containing general cross-platform system informations.
    When to use: When the user asks for the system information or you need to know the system information to perform a task.
    Tags: system, info, platform, os
    """
    info = []
    info.append(f"OS: {platform.system()}")
    info.append(f"OS Version: {platform.release()}")
    info.append(f"Architecture: {platform.machine()}")
    info.append(f"Processor: {platform.processor()}")
    info.append(f"Hostname: {platform.node()}")
    return "\n".join(info)
