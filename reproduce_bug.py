import subprocess
from superior_agent.agent.tools.run_shell import run_shell

def reproduce():
    profile = {"shell": "powershell"}
    workdir = "."
    print("Testing with string timeout...")
    try:
        # Pass timeout as string
        res = run_shell("echo hello", workdir, profile, timeout="60")
        print(f"Result: {res}")
    except Exception as e:
        print(f"Caught at top: {e}")

if __name__ == "__main__":
    reproduce()
