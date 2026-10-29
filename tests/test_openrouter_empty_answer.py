"""Regression tests for hosted OpenRouter empty-completion handling."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "retrieval"))
import deployment_rag as rag  # noqa: E402


class FakeResponse:
    def __init__(self, body, status_code=200, text=""):
        self._body = body
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._body


class OpenRouterEmptyAnswerTests(unittest.TestCase):
    def test_disables_reasoning_and_retries_an_empty_completion(self):
        empty = FakeResponse({
            "choices": [{
                "finish_reason": "length",
                "message": {"content": None, "reasoning": "internal-only; must not be returned"},
            }]
        })
        good = FakeResponse({"choices": [{"finish_reason": "stop", "message": {"content": "Answer grounded in [S1]."}}]})
        with patch.object(rag.requests, "post", side_effect=[empty, good]) as post, patch.object(rag.time, "sleep"):
            result = rag._call_openrouter("test prompt", "test-key", "nvidia/nemotron-3.5-super-vl:free")
        self.assertEqual(result, "Answer grounded in [S1].")
        self.assertEqual(post.call_count, 2)
        first_payload = post.call_args_list[0].kwargs["json"]
        second_payload = post.call_args_list[1].kwargs["json"]
        self.assertEqual(first_payload["reasoning"], {"enabled": False})
        self.assertEqual(second_payload["reasoning"], {"enabled": False})
        self.assertEqual(second_payload["max_tokens"], 1800)
        self.assertNotIn("internal-only", result)

    def test_reports_safe_diagnostics_after_two_empty_completions(self):
        empty = FakeResponse({
            "choices": [{"finish_reason": "length", "message": {"content": None, "reasoning": "do not expose"}}]
        })
        with patch.object(rag.requests, "post", side_effect=[empty, empty]), patch.object(rag.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "empty completion after two attempts") as caught:
                rag._call_openrouter("test prompt", "test-key", "nvidia/nemotron-3.5-super-vl:free")
        self.assertIn("finish_reason=length", str(caught.exception))
        self.assertNotIn("do not expose", str(caught.exception))

    def test_rate_limit_error_is_actionable_and_not_retried(self):
        rate_limited = FakeResponse({"error": {"message": "rate limit"}}, status_code=429, text="rate limit")
        with patch.object(rag.requests, "post", return_value=rate_limited) as post:
            with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                rag._call_openrouter("test prompt", "test-key", "nvidia/nemotron-3.5-super-vl:free")
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
