"""Tests for the agent action JSON parser."""

from agent.llm_agent import parse_action


def test_clean_json():
    raw = '{"action": "click", "target": "search-button"}'
    assert parse_action(raw) == {"action": "click", "target": "search-button"}


def test_with_code_fence():
    raw = '```json\n{"action": "type", "target": "search-input", "text": "USB-C"}\n```'
    result = parse_action(raw)
    assert result["action"] == "type"
    assert result["target"] == "search-input"


def test_with_prose_after():
    raw = '{"action": "click", "target": "add-cable"} I am confident in this action.'
    assert parse_action(raw)["action"] == "click"


def test_invalid_input():
    raw = "I am not sure what to do."
    assert parse_action(raw)["action"] == "invalid"


def test_truncated_json():
    raw = '{"action": "click", "target": "broken'
    assert parse_action(raw)["action"] == "invalid"


def test_done_action():
    raw = '{"action": "done", "reason": "Cart contains target product."}'
    result = parse_action(raw)
    assert result["action"] == "done"
    assert "reason" in result


def test_extra_whitespace():
    raw = '\n\n  {"action": "scroll", "direction": "down"}  \n'
    assert parse_action(raw)["action"] == "scroll"


def test_embedded_newlines_in_target():
    """Real-website page summaries leak \\n inside button labels — when the
    model echoes those back into a JSON string value, strict json.loads fails.
    The forgiving pass should collapse the whitespace and still recover the
    action."""
    raw = '{"action": "click", "target": "Search\nopt\n+"}'
    result = parse_action(raw)
    assert result["action"] == "click"
    assert "Search" in result["target"]


def test_multiline_json_with_code_fence():
    """Code-fenced JSON spanning multiple lines should parse — DOTALL on the
    block extractor handles the newlines between { and }."""
    raw = '```json\n{"action": "done",\n "reason": "found item"}\n```'
    result = parse_action(raw)
    assert result["action"] == "done"
    assert result["reason"] == "found item"
