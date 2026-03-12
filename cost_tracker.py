"""Session cost tracker for LLM token usage."""

from dataclasses import dataclass
import json
import logging


logger = logging.getLogger(__name__)


PRICING = {
    "gpt-4o-mini": {"input": 0.000015, "output": 0.00006},  # per 1K tokens
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (input_tokens * prices["input"] / 1000) + (
        output_tokens * prices["output"] / 1000
    )


@dataclass
class SessionCostTracker:
    session_id: str
    model: str = "gpt-4o-mini"
    budget_usd: float = 0.50
    total_cost_usd: float = 0.0
    call_count: int = 0

    def log_call(
        self,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
    ) -> None:
        cost = calculate_cost(self.model, input_tokens, output_tokens)
        self.total_cost_usd += cost
        self.call_count += 1
        logger.info(
            json.dumps(
                {
                    "event": "llm_call",
                    "session_id": self.session_id,
                    "model": self.model,
                    "cost_usd": cost,
                    "session_total_usd": self.total_cost_usd,
                    "latency_ms": latency_ms,
                    "success": success,
                }
            )
        )

    def check_budget(self) -> bool:
        """Return True if under budget, False if exceeded."""
        return self.total_cost_usd < self.budget_usd
