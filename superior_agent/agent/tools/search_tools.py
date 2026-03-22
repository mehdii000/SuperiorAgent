def search_tools(query: str, registry) -> str:
    """Description: Searches the tool registry for available tools that match the query.
    Args: query: Search term (e.g., 'git', 'web')  registry: The tool registry object
    Returns: A list of matching tools and their descriptions.
    When to use: When you need a tool to accomplish a task but don't have it in your current list of active tools."""
    results = registry.search(query)
    if not results:
        return f"No tools found matching '{query}'"
    
    out = ["Found tools:"]
    for t in results:
        out.append(f"  - {t.name}: {t.description}")
    out.append("Note: The system has automatically activated these tools for you to use in the next round.")
    return "\n".join(out)
