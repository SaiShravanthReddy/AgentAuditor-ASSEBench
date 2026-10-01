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


def _mock_response(status_ok=True, content="judge output", usage=None):
    mock = MagicMock()
    if status_ok:
        mock.raise_for_status.return_value = None
        body = {"choices": [{"message": {"content": content}}]}
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
