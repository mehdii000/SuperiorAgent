import asyncio
import pytest
from superior_agent.agent.brain import Brain, detect_platform
from superior_agent.agent.registry import Registry
from superior_agent.agent.artifact_controller import ArtifactController

class MockLLM:
    async def one_shot(self, *args, **kwargs):
        return {"tier": "moderate", "rationale": "test", "requires_tool": True}
    
    async def stream_chat_with_tools(self, *args, **kwargs):
        # We'll mock this to just return no tool calls to end the loop
        yield type('Event', (), {'type': 'RESPONSE_CHUNK', 'content': 'done'})

def test_brain_initialization():
    registry = Registry()
    brain = Brain(MockLLM(), registry, None, detect_platform())
    assert brain.max_tool_rounds == 20
    assert "increase_max_rounds" in brain.active_tools

def test_increase_max_rounds():
    from superior_agent.agent.tools.increase_max_rounds import increase_max_rounds
    registry = Registry()
    brain = Brain(MockLLM(), registry, None, detect_platform())
    res = increase_max_rounds(brain, increment=5)
    assert brain.max_tool_rounds == 25
    assert "increased max tool rounds by 5" in res

if __name__ == "__main__":
    test_brain_initialization()
    test_increase_max_rounds()
    print("All dynamic round tests passed!")
