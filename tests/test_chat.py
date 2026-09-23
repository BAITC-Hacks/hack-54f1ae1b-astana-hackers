from types import SimpleNamespace
from wind_agent import chat


def test_claude_tool_loop_uses_shared_tool(monkeypatch):
    class FakeTools:
        mode = "production"
        def get_overall_metrics(self):
            return {"actual_available": False}

    class FakeMessages:
        def __init__(self): self.calls = 0
        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="get_overall_metrics", input={}, id="tool_1")])
            assert kwargs["messages"][-1]["content"][0]["type"] == "tool_result"
            assert "actual_available" in kwargs["messages"][-1]["content"][0]["content"]
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="Февральский факт отсутствует.")])

    fake_messages = FakeMessages()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    monkeypatch.setattr(chat, "load_tools", lambda mode: FakeTools())
    monkeypatch.setattr(chat.anthropic, "Anthropic", lambda: SimpleNamespace(messages=fake_messages))
    response = chat.answer("Какая точность за февраль?")
    assert response["tool_calls"] == ["get_overall_metrics"]
    assert "отсутствует" in response["answer"]
