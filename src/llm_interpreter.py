"""
LLM-based market state interpreter with dual backend support.

Supports: Anthropic API (Claude, primary) and local Ollama models (fallback).
Ingests entropy/microstructure features from the physics pipeline, outputs
structured regime assessments with trading recommendations.
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests


# ---------------------------------------------------------------------------
# Numpy-safe JSON encoder
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """Handles numpy int/float types that the default encoder rejects."""

    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "market_interpreter_v1.md"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

VALID_REGIMES = {"calm", "transitional", "crisis_information", "crisis_mechanical"}
VALID_RISK_LEVELS = {"low", "medium", "high", "extreme"}
VALID_CRASH_TYPES = {"information_driven", "mechanical", None}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MarketSnapshot:
    """A single point-in-time market state from the physics pipeline.

    All 16 feature fields correspond to quantities computed in notebooks
    01-06 and the src/ modules. The timestamp identifies the window.
    """

    timestamp: str
    binance_entropy: float
    bybit_entropy: float
    binance_entropy_percentile: float
    net_transfer_entropy: float
    te_leader: str  # "binance" or "bybit"
    te_leader_consecutive_windows: int
    acf_time: float
    acf_time_percentile: float
    acf_trailing_median: float
    realised_vol_30m: float
    order_imbalance: float
    susceptibility: float
    nearest_metastable_level: float
    nearest_well_depth: float
    price: float
    return_5m: float

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(asdict(self), indent=2, cls=_NumpyEncoder)

    @classmethod
    def from_dict(cls, d: dict) -> "MarketSnapshot":
        """Construct from a dictionary, ignoring extra keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class RegimeAssessment:
    """Structured output from the LLM interpreter.

    Fields match the JSON schema enforced in the system prompt.
    """

    timestamp: str
    regime: str  # "calm", "transitional", "crisis_information", "crisis_mechanical"
    confidence: float  # 0.0 to 1.0
    confidence_rationale: str
    crash_type: Optional[str]  # "information_driven", "mechanical", or None
    risk_level: str  # "low", "medium", "high", "extreme"
    conflicting_signals: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    reasoning: str = ""
    physics_summary: str = ""
    key_features_driving_assessment: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(asdict(self), indent=2, cls=_NumpyEncoder)

    @classmethod
    def from_json(cls, text: str) -> "RegimeAssessment":
        """Parse a JSON string into a RegimeAssessment.

        Handles:
        - Clean JSON strings
        - JSON wrapped in markdown code blocks (```json ... ```)
        - Missing optional fields (filled with defaults)
        - Malformed JSON (returns error assessment with confidence=0)

        Parameters
        ----------
        text : str
            Raw text from an LLM response.

        Returns
        -------
        RegimeAssessment
        """
        # Try to extract JSON from markdown code blocks
        cleaned = text.strip()
        code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if code_block:
            cleaned = code_block.group(1).strip()

        try:
            d = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return cls(
                timestamp="parse_error",
                regime="parse_error",
                confidence=0.0,
                confidence_rationale=f"Failed to parse LLM response as JSON: {text[:200]}",
                crash_type=None,
                risk_level="unknown",
                conflicting_signals=[],
                recommended_actions=[],
                reasoning="",
                physics_summary="",
                key_features_driving_assessment=[],
            )

        # Build with defaults for missing fields
        return cls(
            timestamp=d.get("timestamp", "unknown"),
            regime=d.get("regime", "unknown"),
            confidence=float(d.get("confidence", 0.0)),
            confidence_rationale=d.get("confidence_rationale", ""),
            crash_type=d.get("crash_type"),
            risk_level=d.get("risk_level", "unknown"),
            conflicting_signals=d.get("conflicting_signals", []),
            recommended_actions=d.get("recommended_actions", []),
            reasoning=d.get("reasoning", ""),
            physics_summary=d.get("physics_summary", ""),
            key_features_driving_assessment=d.get("key_features_driving_assessment", []),
        )

    def to_markdown(self) -> str:
        """Render as a readable markdown market briefing.

        Used for saving polished reports to ``outputs/llm_reports/``.

        Returns
        -------
        str
            Formatted markdown string.
        """
        crash_str = self.crash_type if self.crash_type else "N/A"
        conflicts = (
            "\n".join(f"- {c}" for c in self.conflicting_signals)
            if self.conflicting_signals
            else "None"
        )
        actions = (
            "\n".join(f"- {a}" for a in self.recommended_actions)
            if self.recommended_actions
            else "None"
        )
        features = (
            "\n".join(f"- {f}" for f in self.key_features_driving_assessment)
            if self.key_features_driving_assessment
            else "None"
        )

        return (
            f"# Market State Assessment — {self.timestamp}\n\n"
            f"**Regime:** {self.regime}  \n"
            f"**Confidence:** {self.confidence:.0%}  \n"
            f"**Risk Level:** {self.risk_level}  \n"
            f"**Crash Type:** {crash_str}\n\n"
            f"## Reasoning\n\n{self.reasoning}\n\n"
            f"## Physics Summary\n\n{self.physics_summary}\n\n"
            f"## Key Features Driving Assessment\n\n{features}\n\n"
            f"## Conflicting Signals\n\n{conflicts}\n\n"
            f"## Confidence Rationale\n\n{self.confidence_rationale}\n\n"
            f"## Recommended Actions\n\n{actions}\n"
        )


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> str:
        """Send a prompt and return the raw text response.

        Parameters
        ----------
        system_prompt : str
            System-level instructions for the LLM.
        user_message : str
            The user message (market snapshot JSON).

        Returns
        -------
        str
            Raw text response from the LLM.
        """
        ...


class AnthropicBackend(LLMBackend):
    """Anthropic API backend (Claude) — primary backend.

    Reads the API key from the ``ANTHROPIC_API_KEY`` environment variable
    by default. The ``anthropic`` package is imported lazily so users who
    only use Ollama never need it installed.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. Defaults to ``os.environ.get("ANTHROPIC_API_KEY")``.
    model : str
        Model identifier (default: claude-sonnet-4-20250514).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_ANTHROPIC_MODEL,
    ):
        import anthropic

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Anthropic API key provided. Set the ANTHROPIC_API_KEY "
                "environment variable or pass api_key= explicitly."
            )
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Send a prompt to the Anthropic API and return the response text."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


class OllamaBackend(LLMBackend):
    """Local Ollama backend — zero-cost fallback.

    Communicates via HTTP to a running Ollama instance. No additional
    Python packages required beyond ``requests`` (already a dependency).

    Parameters
    ----------
    model : str
        Ollama model name (default: llama3.1:8b).
    base_url : str
        Ollama server URL (default: http://localhost:11434).
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Send a prompt to the local Ollama instance and return response text."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "format": "json",
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Market interpreter
# ---------------------------------------------------------------------------


class MarketInterpreter:
    """LLM-powered market state interpreter.

    Wraps an ``LLMBackend`` and a system prompt to assess ``MarketSnapshot``
    objects, returning structured ``RegimeAssessment`` results.

    Parameters
    ----------
    backend : LLMBackend
        The LLM backend to use (AnthropicBackend or OllamaBackend).
    prompt_path : Path, optional
        Path to the system prompt markdown file. Defaults to
        ``prompts/market_interpreter_v1.md`` relative to the repo root.
    """

    def __init__(
        self,
        backend: LLMBackend,
        prompt_path: Optional[Path] = None,
    ):
        self.backend = backend
        path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH

        if not path.exists():
            raise FileNotFoundError(
                f"System prompt not found at {path}. "
                "Ensure prompts/market_interpreter_v1.md exists in the repo root."
            )

        self.system_prompt = path.read_text(encoding="utf-8")

    def assess(self, snapshot: MarketSnapshot) -> RegimeAssessment:
        """Send a single market snapshot to the LLM and parse the response.

        Parameters
        ----------
        snapshot : MarketSnapshot
            Point-in-time market state.

        Returns
        -------
        RegimeAssessment
            Structured regime assessment parsed from the LLM output.
        """
        user_message = snapshot.to_json()
        raw_response = self.backend.complete(self.system_prompt, user_message)
        return RegimeAssessment.from_json(raw_response)

    def assess_batch(
        self,
        snapshots: List[MarketSnapshot],
        delay: float = 0.0,
    ) -> List[RegimeAssessment]:
        """Assess multiple snapshots sequentially.

        Parameters
        ----------
        snapshots : list of MarketSnapshot
            Snapshots to assess.
        delay : float
            Seconds to wait between calls (rate limiting). Default 0.0.

        Returns
        -------
        list of RegimeAssessment
        """
        results = []
        for i, snap in enumerate(snapshots):
            results.append(self.assess(snap))
            if delay > 0 and i < len(snapshots) - 1:
                time.sleep(delay)
        return results

    def evaluate_accuracy(
        self,
        assessments: List[RegimeAssessment],
        ground_truth_regimes: List[str],
    ) -> dict:
        """Compare LLM regime calls against ground truth labels.

        This is a pure computation — no LLM calls are made.

        Parameters
        ----------
        assessments : list of RegimeAssessment
            LLM-generated assessments.
        ground_truth_regimes : list of str
            True regime labels (same length as assessments).

        Returns
        -------
        dict
            Keys: overall_accuracy, per_regime_accuracy, confusion_matrix,
            regime_counts, total.
        """
        if not assessments or not ground_truth_regimes:
            return {
                "overall_accuracy": 0.0,
                "per_regime_accuracy": {},
                "confusion_matrix": {},
                "regime_counts": {},
                "total": 0,
            }

        n = min(len(assessments), len(ground_truth_regimes))
        correct = 0
        regime_correct = {}
        regime_total = {}
        confusion = {}

        for i in range(n):
            predicted = assessments[i].regime
            actual = ground_truth_regimes[i]

            # Count per regime
            regime_total[actual] = regime_total.get(actual, 0) + 1

            # Confusion matrix
            if actual not in confusion:
                confusion[actual] = {}
            confusion[actual][predicted] = confusion[actual].get(predicted, 0) + 1

            if predicted == actual:
                correct += 1
                regime_correct[actual] = regime_correct.get(actual, 0) + 1

        overall_accuracy = correct / n if n > 0 else 0.0

        per_regime_accuracy = {}
        for regime, total in regime_total.items():
            per_regime_accuracy[regime] = regime_correct.get(regime, 0) / total

        return {
            "overall_accuracy": overall_accuracy,
            "per_regime_accuracy": per_regime_accuracy,
            "confusion_matrix": confusion,
            "regime_counts": regime_total,
            "total": n,
        }
