import pandas as pd


REQUIRED_COLUMNS = [
    "repeat_purchase_probability",
]


def prioritize_customers(
    scored_customers: pd.DataFrame,
    capacity_fraction: float,
    customer_id_column: str = "customer_id",
) -> pd.DataFrame:
    """
    Rank customers by model propensity and prioritize the top
    capacity_fraction of the population.

    The customer identifier is treated as decision-layer context
    and is NOT a model feature.
    """

    # ------------------------------------------------------------------
    # Validate capacity
    # ------------------------------------------------------------------

    if not 0 < capacity_fraction <= 1:
        raise ValueError(
            "capacity_fraction must be greater than 0 "
            "and less than or equal to 1."
        )

    # ------------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------------

    required_columns = [
        customer_id_column,
        *REQUIRED_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in scored_customers.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if scored_customers.empty:
        raise ValueError(
            "Cannot prioritize an empty customer population."
        )

    # ------------------------------------------------------------------
    # Validate customer identity
    # ------------------------------------------------------------------

    if scored_customers[
        customer_id_column
    ].isna().any():

        raise ValueError(
            f"{customer_id_column} contains missing values."
        )

    # ------------------------------------------------------------------
    # Validate probabilities
    # ------------------------------------------------------------------

    probabilities = scored_customers[
        "repeat_purchase_probability"
    ]

    if probabilities.isna().any():
        raise ValueError(
            "repeat_purchase_probability contains missing values."
        )

    if not (
        probabilities.between(0, 1).all()
    ):
        raise ValueError(
            "repeat_purchase_probability must be "
            "between 0 and 1."
        )

    # ------------------------------------------------------------------
    # Rank population
    # ------------------------------------------------------------------

    ranked = (
        scored_customers
        .copy()
        .sort_values(
            [
                "repeat_purchase_probability",
                customer_id_column,
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Determine target population
    # ------------------------------------------------------------------

    target_count = max(
        1,
        int(
            len(ranked)
            * capacity_fraction
        ),
    )

    ranked["decision_rank"] = (
        ranked.index + 1
    )

    ranked["priority"] = "standard"

    ranked.loc[
        ranked.index < target_count,
        "priority",
    ] = "prioritize"

    ranked["capacity_fraction"] = (
        capacity_fraction
    )

    ranked["target_population_size"] = (
        target_count
    )

    return ranked