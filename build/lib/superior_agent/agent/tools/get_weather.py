import urllib.request
import urllib.parse
import urllib.error

def get_weather(location: str) -> str:
    """Description: Gets the current weather for a specific location using the wttr.in service.
    Args: location: The city or location name (e.g., 'London', 'San Francisco', 'Paris').
    Returns: The current weather conditions as a string.
    When to use: When the user asks 'what is the weather like in [location]?'.
    """
    loc = urllib.parse.quote(location)
    url = f"https://wttr.in/{loc}?format=3"
    
    try:
        # wttr.in prefers curl User-Agent for plain text responses
        req = urllib.request.Request(url, headers={'User-Agent': 'curl/7.81.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.read().decode('utf-8').strip()
    except Exception as exc:
        return f"Failed to get weather for {location}: {exc}"
