"""Tests for api_telemetry.py - token usage and latency tracking, neither of which was captured
anywhere in this pipeline before (the AI gateway returns a standard "usage" field - confirmed via
live curl calls this session - it was just never read).
"""
import json

from AgentAuditor.tasks.api_telemetry import extract_usage, summarize_calls, write_api_telemetry


class TestExtractUsage:
    def test_standard_openai_shape(self):
        response = {"usage": {"prompt_tokens": 73, "completion_tokens": 31, "total_tokens": 104}}
        assert extract_usage(response) == {
            'prompt_tokens': 73, 'completion_tokens': 31, 'total_tokens': 104,
        }

    def test_missing_usage_returns_none(self):
        assert extract_usage({"choices": []}) is None

    def test_non_dict_response_returns_none(self):
        assert extract_usage("not a dict") is None
        assert extract_usage(None) is None

    def test_malformed_usage_returns_none_not_crash(self):
        assert extract_usage({"usage": {"prompt_tokens": "not a number"}}) is None
        assert extract_usage({"usage": "not a dict"}) is None


class TestSummarizeCalls:
    def test_empty_input(self):
        result = summarize_calls([])
        assert result == {'total_calls': 0, 'successful_calls': 0, 'failed_calls': 0,
                           'latency': None, 'tokens': None}

    def test_all_successful_with_usage(self):
        records = [
            {'elapsed_seconds': 1.0, 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}, 'success': True},
            {'elapsed_seconds': 2.0, 'usage': {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30}, 'success': True},
        ]
        result = summarize_calls(records)
        assert result['total_calls'] == 2
        assert result['successful_calls'] == 2
        assert result['failed_calls'] == 0
        assert result['latency']['mean'] == 1.5
        assert result['tokens']['total_tokens'] == 45
        assert result['tokens']['calls_with_usage'] == 2
        assert result['tokens']['mean_total_tokens_per_call'] == 22.5

    def test_includes_failed_attempts_in_latency_not_just_successful(self):
        """A failed call still took real wall-clock time (and may have consumed tokens before
        erroring) - must count toward latency/retry-overhead visibility, not be silently excluded."""
        records = [
            {'elapsed_seconds': 0.5, 'usage': None, 'success': False},
            {'elapsed_seconds': 1.0, 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}, 'success': True},
        ]
        result = summarize_calls(records)
        assert result['total_calls'] == 2
        assert result['failed_calls'] == 1
        assert result['latency']['mean'] == 0.75  # both attempts count toward latency
        assert result['tokens']['total_tokens'] == 15  # only the successful call had usage

    def test_percentiles_use_nearest_rank_no_crash_on_small_n(self):
        records = [{'elapsed_seconds': float(i), 'usage': None, 'success': True} for i in range(1, 11)]
        result = summarize_calls(records)
        assert result['latency']['p50'] in (5.0, 6.0)  # nearest-rank, either side is reasonable
        assert result['latency']['max'] == 10.0
        assert result['latency']['min'] == 1.0

    def test_no_usage_anywhere_returns_none_tokens_not_crash(self):
        records = [{'elapsed_seconds': 1.0, 'usage': None, 'success': True}]
        result = summarize_calls(records)
        assert result['tokens'] is None
        assert result['latency'] is not None  # latency still computed even without token data


class TestWriteApiTelemetry:
    def test_writes_stage_specific_sidecar_file(self, tmp_path):
        metrics = {'total_calls': 5, 'successful_calls': 5, 'failed_calls': 0}
        path = write_api_telemetry(str(tmp_path), 'infer', metrics)
        assert path == str(tmp_path / 'api_telemetry_infer.json')
        with open(path) as f:
            assert json.load(f) == metrics
