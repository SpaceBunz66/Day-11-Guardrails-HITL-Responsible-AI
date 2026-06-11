"""
Lab 11 — Part 4: Human-in-the-Loop Design
  TODO 12: Confidence Router
  TODO 13: Design 3 HITL decision points
"""
from dataclasses import dataclass


HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
]
HIGH_RISK_ACTIONS_SET = frozenset(HIGH_RISK_ACTIONS)


@dataclass
class RoutingDecision:
    """Result of the confidence router."""
    action: str          # "auto_send", "queue_review", "escalate"
    confidence: float
    reason: str
    priority: str        # "low", "normal", "high"
    requires_human: bool


class ConfidenceRouter:
    """Route agent responses based on confidence and risk level.

    Thresholds:
        HIGH:   confidence >= 0.9 -> auto-send
        MEDIUM: 0.7 <= confidence < 0.9 -> queue for review
        LOW:    confidence < 0.7 -> escalate to human

    High-risk actions always escalate regardless of confidence.
    """

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(self, response: str, confidence: float,
              action_type: str = "general") -> RoutingDecision:
        """Route a response based on confidence score and action type.

        Args:
            response: The agent's response text
            confidence: Confidence score between 0.0 and 1.0
            action_type: Type of action (e.g., "general", "transfer_money")

        Returns:
            RoutingDecision with routing action and metadata
        """
        if action_type in HIGH_RISK_ACTIONS_SET:
            return RoutingDecision(
                action="escalate",
                confidence=confidence,
                reason=f"High-risk action: {action_type}",
                priority="high",
                requires_human=True,
            )

        if confidence >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                confidence=confidence,
                reason="High confidence",
                priority="low",
                requires_human=False,
            )

        if confidence >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                confidence=confidence,
                reason="Medium confidence - needs review",
                priority="normal",
                requires_human=True,
            )

        return RoutingDecision(
            action="escalate",
            confidence=confidence,
            reason="Low confidence - escalating",
            priority="high",
            requires_human=True,
        )


hitl_decision_points = [
    {
        "id": 1,
        "name": "High-value transaction approval",
        "trigger": "A customer requests a money movement above policy limits or matching high-risk patterns.",
        "hitl_model": "human-in-the-loop",
        "context_needed": "Customer identity checks, transaction amount, destination account, recent account activity, fraud signals, and customer-stated purpose.",
        "example": "A customer asks the agent to wire $50,000 to a new overseas beneficiary.",
    },
    {
        "id": 2,
        "name": "Account closure or restriction",
        "trigger": "The agent recommends closing, freezing, or limiting an account based on policy, risk, or customer request.",
        "hitl_model": "human-on-the-loop",
        "context_needed": "Reason for the recommendation, account status, open disputes, pending transactions, regulatory flags, and prior customer communications.",
        "example": "An automated risk review recommends freezing an account after several unusual login and transfer attempts.",
    },
    {
        "id": 3,
        "name": "Disputed charge resolution",
        "trigger": "Evidence is mixed or model confidence is too close to the approve/deny threshold for a dispute.",
        "hitl_model": "human-as-tiebreaker",
        "context_needed": "Merchant details, transaction metadata, customer dispute reason, supporting documents, prior dispute history, and model rationale.",
        "example": "A customer disputes a hotel charge, but location data and merchant records only partially support the claim.",
    },
]


# ============================================================
# Quick tests
# ============================================================

def test_confidence_router():
    """Test ConfidenceRouter with sample scenarios."""
    router = ConfidenceRouter()

    test_cases = [
        ("Balance inquiry", 0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("Ambiguous request", 0.55, "general"),
        ("Transfer $50,000", 0.98, "transfer_money"),
        ("Close my account", 0.91, "close_account"),
    ]

    print("Testing ConfidenceRouter:")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Conf':<6} {'Action Type':<18} {'Decision':<15} {'Priority':<10} {'Human?'}")
    print("-" * 80)

    for scenario, conf, action_type in test_cases:
        decision = router.route(scenario, conf, action_type)
        print(
            f"{scenario:<25} {conf:<6.2f} {action_type:<18} "
            f"{decision.action:<15} {decision.priority:<10} "
            f"{'Yes' if decision.requires_human else 'No'}"
        )

    print("=" * 80)


def test_hitl_points():
    """Display HITL decision points."""
    print("\nHITL Decision Points:")
    print("=" * 60)
    for point in hitl_decision_points:
        print(f"\n  Decision Point #{point['id']}: {point['name']}")
        print(f"    Trigger:  {point['trigger']}")
        print(f"    Model:    {point['hitl_model']}")
        print(f"    Context:  {point['context_needed']}")
        print(f"    Example:  {point['example']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_confidence_router()
    test_hitl_points()
