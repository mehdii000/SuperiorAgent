import textwrap
from pathlib import Path
import pytest
from superior_agent.agent.registry import Registry
from superior_agent.agent.tools.edit_file import edit_file

def test_semantic_search_with_tags(tmp_path):
    (tmp_path / "search_tool.py").write_text(textwrap.dedent("""
        def search_tool(q: str):
            \"\"\"Description: A search tool.
            Tags: internet, web, search
            \"\"\"
            return f"Searching for {q}"
    """))
    
    reg = Registry()
    reg.discover(tmp_path)
    
    # Test multi-word query
    results = reg.search("search internet")
    assert len(results) == 1
    assert results[0].name == "search_tool"
    
    # Test partial word in tags
    results = reg.search("web")
    assert len(results) == 1
    
    # No match
    results = reg.search("weather")
    assert len(results) == 0

def test_edit_file_line_range(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line 1\nline 2\nline 3\nline 4\n")
    
    # Replace lines 2-3
    res = edit_file(str(f), new_text="NEW LINE 2\nNEW LINE 3", start_line=2, end_line=3, workdir=str(tmp_path))
    assert "Successfully edited" in res
    
    content = f.read_text()
    expected = "line 1\nNEW LINE 2\nNEW LINE 3\nline 4\n"
    assert content == expected

def test_edit_file_line_range_with_safety_check(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line 1\nline 2\nline 3\n")
    
    # Correct old_text
    res = edit_file(str(f), old_text="line 2\nline 3\n", new_text="CHANGED", start_line=2, end_line=3, workdir=str(tmp_path))
    assert "Successfully edited" in res
    
    # Incorrect old_text
    res = edit_file(str(f), old_text="WRONG", new_text="CHANGED", start_line=1, end_line=1, workdir=str(tmp_path))
    assert "Error: The provided old_text does not match" in res

def test_edit_file_fallback_to_text(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("apple\nbanana\ncherry\n")
    
    res = edit_file(str(f), old_text="banana", new_text="ORANGE", workdir=str(tmp_path))
    assert "Successfully edited" in res
    assert "ORANGE" in f.read_text()

def test_write_file_identity_check(tmp_path):
    from superior_agent.agent.tools.write_file import write_file
    f = tmp_path / "identity.txt"
    f.write_text("same content")
    
    res = write_file("identity.txt", "same content", str(tmp_path))
    assert "Skipped: Content" in res

def test_write_file_atomic(tmp_path):
    from superior_agent.agent.tools.write_file import write_file
    res = write_file("atomic.txt", "new content", str(tmp_path))
    assert "Successfully wrote" in res
    assert (tmp_path / "atomic.txt").read_text() == "new content"

def test_run_shell_truncation(tmp_path):
    from superior_agent.agent.tools.run_shell import run_shell
    # Mocking platform_profile for Windows/Linux
    profile = {"shell": "powershell"} 
    
    # Large output
    large_cmd = "Write-Output ('A' * 11000)"
    res = run_shell(large_cmd, str(tmp_path), profile, max_output_length=100)
    assert "Output Truncated" in res
    assert len(res) < 500 # Should be truncated

def test_run_shell_timeout(tmp_path):
    from superior_agent.agent.tools.run_shell import run_shell
    profile = {"shell": "powershell"}
    
    res = run_shell("Start-Sleep -Seconds 2", str(tmp_path), profile, timeout=1)
    assert "timed out after 1 seconds" in res
