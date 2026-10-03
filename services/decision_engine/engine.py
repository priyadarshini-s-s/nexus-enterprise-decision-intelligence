from dataclasses import dataclass
from typing import Literal


# =============================================================================
# NEXUS — Decision Engine
# =============================================================================
#
# Important:
# This engine does NOT claim that an intervention causes a purchase.
#
# It combines predictive propensity with explicit business constraints.
#
# Causal/uplift evidence will be incorporated later when the NEXUS
# experimentation module is implemented.
# =============================================================================


Decision = Literal[
    "prioritize",
    "standard",
    "deprioritize",
]


@dataclass(frozen=True)
class RetentionDecisionInput:
    """
    Inputs required by the retention decision policy.

    propensity:
        Predicted probability of repeat_purchase_90d.

    customer_value:
        Observed customer monetary value supplied by the upstream
        Customer 360 / feature layer.

    capacity_percentile:
        Optional operational capacity boundary.

        Example:
            0.10 means the business currently has capacity to prioritize
            approximately the top 10% of scored customers.
    """

    propensity: float
    customer_value: float
    capacity_percentile: float = 0.10


@dataclass(frozen=True)
class RetentionDecision:
    decision: Decision
    reason: str
    propensity: float
    customer_value: float


def make_retention_decision(
    inputs: RetentionDecisionInput,
) -> RetentionDecision:

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    if not 0.0 <= inputs.propensity <= 1.0:
        raise ValueError(
            "Propensity must be between 0 and 1."
        )

    if inputs.customer_value < 0:
        raise ValueError(
            "Customer value cannot be negative."
        )

    if not 0.0 < inputs.capacity_percentile <= 1.0:
        raise ValueError(
            "capacity_percentile must be > 0 and <= 1."
        )

    # -------------------------------------------------------------------------
    # Current policy
    # -------------------------------------------------------------------------
    #
    # At this stage we intentionally avoid inventing an arbitrary probability
    # threshold such as:
    #
    #     probability > 0.5 => contact
    #
    # The actual positive prevalence is below 1%, so such a threshold would
    # have no operational meaning.
    #
    # The current engine therefore uses an explicit propensity threshold
    # derived from the capacity policy.
    #
    # The exact threshold will later be replaced by a percentile/ranking
    # operation over the scored customer population.
    # -------------------------------------------------------------------------

    if inputs.propensity >= 0.10:

        return RetentionDecision(
            decision="prioritize",
            reason=(
                "Propensity exceeds the current decision-policy "
                "threshold."
            ),
            propensity=inputs.propensity,
            customer_value=inputs.customer_value,
        )

    if inputs.propensity >= 0.01:

        return RetentionDecision(
            decision="standard",
            reason=(
                "Customer has measurable predicted repeat-purchase "
                "propensity but does not exceed the current "
                "priority threshold."
            ),
            propensity=inputs.propensity,
            customer_value=inputs.customer_value,
        )

    return RetentionDecision(
        decision="deprioritize",
        reason=(
            "Predicted repeat-purchase propensity is below the "
            "current decision-policy threshold."
        ),
        propensity=inputs.propensity,
        customer_value=inputs.customer_value,
    )