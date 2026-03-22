import datetime

def get_current_time() -> str:
    """Description: Retrieves the current local date and time.
    Returns: The current date and time as a string.
    When to use: When the user asks for the current time or date, or you need to timestamp something.
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
