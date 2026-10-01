"""Tests for LLMHandler.call_llm_api()'s api_telemetry wiring - verifies the actual integration
(requests.post mocked, not call_llm_api itself mocked), since test_infer_process_json_file.py's
existing tests patch call_llm_api wholesale and never exercise this code path.
"""
from unittest.mock import MagicMock, patch

from AgentAuditor.tasks.infer import GPTConfig, LLMHandler


def _config():
    config = GPTConfig()
    config.MAX_RETRIES = 2
    config.RETRY_DELAY = 0  # don't actually sleep in tests
    return config


def _mock_response(status_ok=True, content="judge output", usage=None, reasoning_content=None):
    mock = MagicMock()
    if status_ok:
        mock.raise_for_status.return_value = None
        message = {"content": content}
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        body = {"choices": [{"message": message}]}
        if usage is not None:
            body["usage"] = usage
        mock.json.return_value = body
    else:
        mock.raise_for_status.side_effect = Exception("HTTP error")
    return mock


class TestCallLlmApiTelemetry:
    def test_successful_call_records_usage_and_latency(self):
        handler = LLMHandler(_config())
        usage = {"prompt_tokens": 73, "completion_tokens": 31, "total_tokens": 104}
        with patch("AgentAuditor.tasks.infer.requests.post", return_value=_mock_response(usage=usage)):
            result = handler.call_llm_api("some prompt", item_id="item-1")

        assert result == "judge output"
        assert len(handler.call_records) == 1
        record = handler.call_records[0]
        assert record['success'] is True
        assert record['usage'] == {'prompt_tokens': 73, 'completion_tokens': 31, 'total_tokens': 104}
        assert record['elapsed_seconds'] >= 0

    def test_failed_call_records_failure_without_crashing(self):
        handler = LLMHandler(_config())
        with patch("AgentAuditor.tasks.infer.requests.post", return_value=_mock_response(status_ok=False)):
            result = handler.call_llm_api("some prompt", item_id="item-1")

        assert result is None
        # MAX_RETRIES=2, every attempt fails -> 2 records, both failed
        assert len(handler.call_records) == 2
        assert all(r['success'] is False for r in handler.call_records)
        assert all(r['usage'] is None for r in handler.call_records)

    def test_response_with_no_usage_field_records_none_not_crash(self):
        handler = LLMHandler(_config())
        with patch("AgentAuditor.tasks.infer.requests.post", return_value=_mock_response(usage=None)):
            handler.call_llm_api("some prompt", item_id="item-1")

        assert handler.call_records[0]['usage'] is None
        assert handler.call_records[0]['success'] is True

    def test_records_accumulate_across_multiple_calls_on_same_handler(self):
        """process_json_file() creates one LLMHandler and reuses it for the whole dataset - records
        must accumulate across calls, not reset each time."""
        handler = LLMHandler(_config())
        with patch("AgentAuditor.tasks.infer.requests.post", return_value=_mock_response()):
            handler.call_llm_api("prompt 1", item_id="item-1")
            handler.call_llm_api("prompt 2", item_id="item-2")

        assert len(handler.call_records) == 2


class TestReasoningContentFallback:
    """Regression guard for a real incident (2026-10): some responses from this gateway (confirmed
    on gpt-oss-20b, for longer/more complex prompts) put the model's actual answer in
    reasoning_content instead of content, leaving content literally null despite a complete, valid
    JSON verdict being present. The old code only ever read content, so a perfectly good answer
    was silently discarded and the item got endlessly marked "failed" and retried for no real
    reason - confirmed by dumping a raw API response for a real production item
    (harmless-MT_App-28) that was wrongly stuck in failed.json: HTTP 200, finish_reason "stop",
    content: null, reasoning_content: the full real verdict JSON.
    """

    def test_null_content_falls_back_to_reasoning_content(self):
        handler = LLMHandler(_config())
        real_verdict = '{\n  "label": 0,\n  "confidence": 0.99,\n  "explanation": "..."\n}'
        mock = _mock_response(content=None, reasoning_content=real_verdict)
        with patch("AgentAuditor.tasks.infer.requests.post", return_value=mock):
            result = handler.call_llm_api("some long prompt", item_id="harmless-MT_App-28")

        assert result == real_verdict
        assert handler.call_records[0]['success'] is True

    def test_empty_string_content_also_falls_back(self):
        """Not just None - an empty string content is equally useless and should also trigger the
        fallback, since both represent "no real answer in the expected field"."""
        handler = LLMHandler(_config())
        mock = _mock_response(content="", reasoning_content='{"label": 1}')
        with patch("AgentAuditor.tasks.infer.requests.post", return_value=mock):
            result = handler.call_llm_api("some prompt", item_id="item-1")

        assert result == '{"label": 1}'

    def test_populated_content_is_used_directly_not_overridden(self):
        """The normal, common case must be unaffected: when content is genuinely populated, use it
        as-is, even if reasoning_content is also present (don't prefer reasoning_content blindly)."""
        handler = LLMHandler(_config())
        mock = _mock_response(content="real content here", reasoning_content="should not be used")
        with patch("AgentAuditor.tasks.infer.requests.post", return_value=mock):
            result = handler.call_llm_api("some prompt", item_id="item-1")

        assert result == "real content here"

    def test_both_content_and_reasoning_content_missing_returns_none(self):
        """If there's genuinely no answer anywhere in the response, the function must still return
        None (and the caller's existing "treat None as failure" logic still applies) rather than
        raising or returning something misleading."""
        handler = LLMHandler(_config())
        mock = _mock_response(content=None, reasoning_content=None)
        with patch("AgentAuditor.tasks.infer.requests.post", return_value=mock):
            result = handler.call_llm_api("some prompt", item_id="item-1")

        assert result is None
