"""Tests for src/llm_interpreter.py — LLM market state interpreter.

Unit tests (no LLM needed) verify serialisation, parsing, and accuracy
computation. Integration tests (marked @pytest.mark.integration) make
actual LLM calls and require a running backend.
"""

import json

import pytest

from src.llm_interpreter import (
    AnthropicBackend,
    LLMBackend,
    MarketInterpreter,
    MarketSnapshot,
    OllamaBackend,
    RegimeAssessment,
)


# ---------------------------------------------------------------------------
# Unit tests: MarketSnapshot
# ---------------------------------------------------------------------------


class TestMarketSnapshot:
    """Tests for MarketSnapshot dataclass serialisation."""

    def test_to_json_roundtrip(self, sample_market_snapshot):
        """Serialise to JSON and reconstruct; all fields match."""
        json_str = sample_market_snapshot.to_json()
        d = json.loads(json_str)
        rebuilt = MarketSnapshot.from_dict(d)

        assert rebuilt.timestamp == sample_market_snapshot.timestamp
        assert rebuilt.binance_entropy == sample_market_snapshot.binance_entropy
        assert rebuilt.price == sample_market_snapshot.price
        assert rebuilt.return_5m == sample_market_snapshot.return_5m

    def test_all_fields_present_in_json(self, sample_market_snapshot):
        """JSON output contains all 17 fields."""
        d = json.loads(sample_market_snapshot.to_json())
        expected_fields = {
            "timestamp", "binance_entropy", "bybit_entropy",
            "binance_entropy_percentile", "net_transfer_entropy",
            "te_leader", "te_leader_consecutive_windows", "acf_time",
            "acf_time_percentile", "acf_trailing_median",
            "realised_vol_30m", "order_imbalance", "susceptibility",
            "nearest_metastable_level", "nearest_well_depth",
            "price", "return_5m",
        }
        assert set(d.keys()) == expected_fields

    def test_from_dict_ignores_extra_keys(self, sample_market_snapshot):
        """Extra keys in the dict are silently ignored."""
        d = json.loads(sample_market_snapshot.to_json())
        d["extra_field"] = 999
        rebuilt = MarketSnapshot.from_dict(d)
        assert rebuilt.timestamp == sample_market_snapshot.timestamp

    def test_field_types(self, sample_market_snapshot):
        """Verify critical field types."""
        assert isinstance(sample_market_snapshot.timestamp, str)
        assert isinstance(sample_market_snapshot.binance_entropy, float)
        assert isinstance(sample_market_snapshot.te_leader, str)
        assert isinstance(sample_market_snapshot.te_leader_consecutive_windows, int)


# ---------------------------------------------------------------------------
# Unit tests: RegimeAssessment
# ---------------------------------------------------------------------------


VALID_ASSESSMENT_JSON = json.dumps({
    "timestamp": "2026-01-31T14:30:00Z",
    "regime": "crisis_information",
    "confidence": 0.92,
    "confidence_rationale": "All indicators align.",
    "crash_type": "information_driven",
    "risk_level": "extreme",
    "conflicting_signals": [],
    "recommended_actions": ["Reduce exposure"],
    "reasoning": "Entropy collapsed.",
    "physics_summary": "Phase transition detected.",
    "key_features_driving_assessment": ["entropy: 2nd percentile"],
})


class TestRegimeAssessment:
    """Tests for RegimeAssessment parsing and serialisation."""

    def test_from_json_valid(self):
        """Parse a well-formed JSON string."""
        ra = RegimeAssessment.from_json(VALID_ASSESSMENT_JSON)
        assert ra.regime == "crisis_information"
        assert ra.confidence == 0.92
        assert ra.crash_type == "information_driven"
        assert ra.risk_level == "extreme"
        assert len(ra.recommended_actions) == 1

    def test_from_json_null_crash_type(self):
        """crash_type=null parses to None."""
        d = json.loads(VALID_ASSESSMENT_JSON)
        d["crash_type"] = None
        ra = RegimeAssessment.from_json(json.dumps(d))
        assert ra.crash_type is None

    def test_from_json_missing_optional_fields(self):
        """Missing list/string fields get defaults."""
        minimal = json.dumps({
            "timestamp": "2026-02-03T10:00:00Z",
            "regime": "calm",
            "confidence": 0.85,
            "confidence_rationale": "All calm.",
            "risk_level": "low",
        })
        ra = RegimeAssessment.from_json(minimal)
        assert ra.regime == "calm"
        assert ra.conflicting_signals == []
        assert ra.recommended_actions == []
        assert ra.reasoning == ""
        assert ra.crash_type is None

    def test_from_json_malformed_returns_error(self):
        """Invalid JSON returns a parse-error assessment."""
        ra = RegimeAssessment.from_json("this is not json {{{")
        assert ra.regime == "parse_error"
        assert ra.confidence == 0.0
        assert "Failed to parse" in ra.confidence_rationale

    def test_from_json_wrapped_in_code_block(self):
        """JSON wrapped in ```json ... ``` is parsed correctly."""
        wrapped = f"```json\n{VALID_ASSESSMENT_JSON}\n```"
        ra = RegimeAssessment.from_json(wrapped)
        assert ra.regime == "crisis_information"
        assert ra.confidence == 0.92

    def test_from_json_wrapped_in_plain_code_block(self):
        """JSON wrapped in ``` ... ``` (no language tag) is parsed."""
        wrapped = f"```\n{VALID_ASSESSMENT_JSON}\n```"
        ra = RegimeAssessment.from_json(wrapped)
        assert ra.regime == "crisis_information"

    def test_to_json_roundtrip(self):
        """to_json -> from_json produces equivalent assessment."""
        ra = RegimeAssessment.from_json(VALID_ASSESSMENT_JSON)
        json_str = ra.to_json()
        ra2 = RegimeAssessment.from_json(json_str)
        assert ra2.regime == ra.regime
        assert ra2.confidence == ra.confidence
        assert ra2.crash_type == ra.crash_type

    def test_to_markdown_contains_regime(self):
        """Markdown output includes the regime label."""
        ra = RegimeAssessment.from_json(VALID_ASSESSMENT_JSON)
        md = ra.to_markdown()
        assert "crisis_information" in md
        assert "Market State Assessment" in md

    def test_to_markdown_contains_actions(self):
        """Markdown output includes recommended actions."""
        ra = RegimeAssessment.from_json(VALID_ASSESSMENT_JSON)
        md = ra.to_markdown()
        assert "Reduce exposure" in md


# ---------------------------------------------------------------------------
# Unit tests: evaluate_accuracy
# ---------------------------------------------------------------------------


class TestEvaluateAccuracy:
    """Tests for MarketInterpreter.evaluate_accuracy (no LLM needed).

    We call evaluate_accuracy as a standalone method by creating a
    minimal interpreter with a mock backend.
    """

    def _make_interpreter(self):
        """Create a MarketInterpreter with a dummy backend."""

        class DummyBackend(LLMBackend):
            def complete(self, system_prompt, user_message):
                return "{}"

        return MarketInterpreter(
            backend=DummyBackend(),
            prompt_path=None,
        )

    def _make_assessment(self, regime):
        return RegimeAssessment(
            timestamp="t",
            regime=regime,
            confidence=0.8,
            confidence_rationale="",
            crash_type=None,
            risk_level="low",
        )

    def test_perfect_accuracy(self):
        """All correct -> accuracy = 1.0."""
        interp = self._make_interpreter()
        assessments = [self._make_assessment(r) for r in ["calm", "calm", "crisis_information"]]
        gt = ["calm", "calm", "crisis_information"]
        result = interp.evaluate_accuracy(assessments, gt)
        assert result["overall_accuracy"] == 1.0
        assert result["total"] == 3

    def test_zero_accuracy(self):
        """All wrong -> accuracy = 0.0."""
        interp = self._make_interpreter()
        assessments = [self._make_assessment("calm")] * 3
        gt = ["crisis_information", "crisis_mechanical", "transitional"]
        result = interp.evaluate_accuracy(assessments, gt)
        assert result["overall_accuracy"] == 0.0

    def test_partial_accuracy(self):
        """3 of 5 correct -> accuracy = 0.6."""
        interp = self._make_interpreter()
        assessments = [
            self._make_assessment("calm"),          # correct
            self._make_assessment("calm"),           # wrong (gt=transitional)
            self._make_assessment("crisis_information"),  # correct
            self._make_assessment("transitional"),   # correct
            self._make_assessment("crisis_mechanical"),  # wrong (gt=calm)
        ]
        gt = ["calm", "transitional", "crisis_information", "transitional", "calm"]
        result = interp.evaluate_accuracy(assessments, gt)
        assert abs(result["overall_accuracy"] - 0.6) < 1e-9

    def test_confusion_matrix_structure(self):
        """Confusion matrix keys sum to total count."""
        interp = self._make_interpreter()
        assessments = [
            self._make_assessment("calm"),
            self._make_assessment("crisis_information"),
        ]
        gt = ["calm", "calm"]
        result = interp.evaluate_accuracy(assessments, gt)
        cm = result["confusion_matrix"]
        total = sum(v for inner in cm.values() for v in inner.values())
        assert total == 2
        assert cm["calm"]["calm"] == 1
        assert cm["calm"]["crisis_information"] == 1

    def test_empty_input(self):
        """Empty lists -> accuracy = 0.0."""
        interp = self._make_interpreter()
        result = interp.evaluate_accuracy([], [])
        assert result["overall_accuracy"] == 0.0
        assert result["total"] == 0


# ---------------------------------------------------------------------------
# Unit tests: Backend acceptance
# ---------------------------------------------------------------------------


class TestBackendAcceptance:
    """Verify class hierarchy and backend polymorphism."""

    def test_ollama_is_llm_backend(self):
        """OllamaBackend is a subclass of LLMBackend."""
        assert issubclass(OllamaBackend, LLMBackend)

    def test_anthropic_is_llm_backend(self):
        """AnthropicBackend is a subclass of LLMBackend."""
        assert issubclass(AnthropicBackend, LLMBackend)

    def test_interpreter_accepts_any_backend(self):
        """MarketInterpreter works with any LLMBackend subclass."""

        class MockBackend(LLMBackend):
            def complete(self, system_prompt, user_message):
                return '{"regime": "calm"}'

        interp = MarketInterpreter(backend=MockBackend())
        assert interp.backend is not None

    def test_interpreter_raises_on_missing_prompt(self, tmp_path):
        """FileNotFoundError if prompt file does not exist."""

        class MockBackend(LLMBackend):
            def complete(self, system_prompt, user_message):
                return "{}"

        with pytest.raises(FileNotFoundError):
            MarketInterpreter(
                backend=MockBackend(),
                prompt_path=tmp_path / "nonexistent.md",
            )


# ---------------------------------------------------------------------------
# Integration tests (require a running LLM backend)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLLMIntegration:
    """Integration tests that make actual LLM calls.

    Run with: pytest tests/ -v -m integration --backend ollama
    Skipped by default when running: pytest tests/ -v
    """

    def _get_backend(self, backend_name):
        """Instantiate the requested backend, skip if unavailable."""
        if backend_name == "anthropic":
            import os
            if not os.environ.get("ANTHROPIC_API_KEY"):
                pytest.skip("ANTHROPIC_API_KEY not set")
            return AnthropicBackend()
        else:
            import requests
            try:
                resp = requests.get("http://localhost:11434/api/tags", timeout=5)
                resp.raise_for_status()
            except Exception:
                pytest.skip("Ollama not running at localhost:11434")
            return OllamaBackend()

    def test_single_calm_snapshot(self, backend_name, sample_market_snapshot):
        """Send a calm snapshot; verify output parses to valid RegimeAssessment."""
        backend = self._get_backend(backend_name)
        interp = MarketInterpreter(backend=backend)
        result = interp.assess(sample_market_snapshot)

        assert isinstance(result, RegimeAssessment)
        assert result.regime != "parse_error"
        assert 0.0 <= result.confidence <= 1.0
        assert result.timestamp == sample_market_snapshot.timestamp

    def test_single_crisis_snapshot(self, backend_name, sample_crisis_snapshot):
        """Send a crisis snapshot; verify regime is a crisis type."""
        backend = self._get_backend(backend_name)
        interp = MarketInterpreter(backend=backend)
        result = interp.assess(sample_crisis_snapshot)

        assert isinstance(result, RegimeAssessment)
        assert result.regime != "parse_error"
        # The LLM should recognise this as some form of crisis or at least not calm
        assert result.regime in {
            "crisis_information", "crisis_mechanical", "transitional"
        }

    def test_contradictory_signals_lower_confidence(self, backend_name):
        """Snapshot with contradictory features should produce low confidence."""
        backend = self._get_backend(backend_name)
        interp = MarketInterpreter(backend=backend)

        # Contradictory: low entropy (crisis) but low ACF time (calm)
        contradictory = MarketSnapshot(
            timestamp="2026-02-02T08:15:00Z",
            binance_entropy=0.72,
            bybit_entropy=0.80,
            binance_entropy_percentile=8.0,
            net_transfer_entropy=0.005,
            te_leader="binance",
            te_leader_consecutive_windows=3,
            acf_time=0.9,
            acf_time_percentile=20.0,
            acf_trailing_median=1.4,
            realised_vol_30m=0.003,
            order_imbalance=-0.20,
            susceptibility=0.002,
            nearest_metastable_level=82000.0,
            nearest_well_depth=5.5,
            price=82100.0,
            return_5m=-0.001,
        )

        result = interp.assess(contradictory)
        assert isinstance(result, RegimeAssessment)
        assert result.regime != "parse_error"
        # With contradictory signals, confidence should be moderate to low
        assert result.confidence < 0.8

    def test_output_schema_always_valid(self, backend_name, sample_market_snapshot):
        """Output JSON always has all required fields."""
        backend = self._get_backend(backend_name)
        interp = MarketInterpreter(backend=backend)
        result = interp.assess(sample_market_snapshot)

        # Parse back to JSON and verify all fields present
        d = json.loads(result.to_json())
        required = {
            "timestamp", "regime", "confidence", "confidence_rationale",
            "crash_type", "risk_level", "conflicting_signals",
            "recommended_actions", "reasoning", "physics_summary",
            "key_features_driving_assessment",
        }
        assert required.issubset(set(d.keys()))

    def test_batch_assessment(
        self, backend_name, sample_market_snapshot, sample_crisis_snapshot
    ):
        """Batch of 2 snapshots returns 2 valid assessments."""
        backend = self._get_backend(backend_name)
        interp = MarketInterpreter(backend=backend)
        results = interp.assess_batch(
            [sample_market_snapshot, sample_crisis_snapshot], delay=0.5
        )
        assert len(results) == 2
        assert all(isinstance(r, RegimeAssessment) for r in results)
        assert all(r.regime != "parse_error" for r in results)
