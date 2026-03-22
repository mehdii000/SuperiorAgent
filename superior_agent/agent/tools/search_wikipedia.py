import json
import urllib.request
import urllib.parse
import urllib.error

def search_wikipedia(query: str) -> str:
    """Description: Searches Wikipedia for a given query and returns a summary of the topic.
    Args: query: The topic to search for on Wikipedia.
    Returns: A short summary of the Wikipedia article.
    When to use: When the user asks for factual information, history, or a summary of a specific topic, person, or event.
    """
    q = urllib.parse.quote(query)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'SuperiorAgent/1.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode('utf-8'))
            extract = data.get('extract')
            if extract:
                return extract
            return "No summary found for this topic."
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"No Wikipedia article found for '{query}'."
        return f"Wikipedia search failed: HTTP {e.code}"
    except Exception as exc:
        return f"Failed to search Wikipedia for '{query}': {exc}"
