import pytest
from code_assistant.llm.qrok_qwen_llm import GroqQwenLLM


@pytest.fixture
def llm():
    return GroqQwenLLM()


def test_normalize_results(llm):
    fake_response = {
        "ids": [["123"]],
        "metadatas": [[{
            "file_path": "/path/example.ts",
            "name": "testFunction",
            "type": "method_definition",
            "start_line": 10,
            "end_line": 20
        }]],
        "documents": [[
            "function testFunction() { return 1; }"
        ]]
    }

    result = llm._normalize_results(fake_response)

    assert isinstance(result, str)
    assert "example.ts" in result
    assert "testFunction" in result
    assert "10–20" in result
    assert "function testFunction()" in result
