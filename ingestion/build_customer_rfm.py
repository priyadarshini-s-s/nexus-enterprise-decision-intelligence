from pathlib import Path
import pandas as pd


GOLD_DIR = Path("data/gold/olist")


def main():

    print("=" * 80)
    print("NEXUS — Customer RFM Gold Model")
    print("=" * 80)

    customer_360 = pd.read_parquet(
        GOLD_DIR / "customer_360.parquet"
    )

    # ---------------------------------------------------------------
    # Analysis cutoff
    # ---------------------------------------------------------------

    analysis_cutoff = (
        customer_360["last_order_date"].max()
    )

    print(
        f"\nAnalysis cutoff: {analysis_cutoff}"
    )

    # ---------------------------------------------------------------
    # RFM construction
    # ---------------------------------------------------------------

    rfm = customer_360[
        [
            "customer_unique_id",
            "first_order_date",
            "last_order_date",
            "total_orders",
            "total_payment_value",
            "average_order_value",
            "is_repeat_customer",
            "payment_observation_status",
        ]
    ].copy()

    rfm["recency_days"] = (
        analysis_cutoff
        - rfm["last_order_date"]
    ).dt.days

    rfm["frequency"] = rfm["total_orders"]

    rfm["monetary"] = rfm["total_payment_value"]

    rfm["analysis_cutoff_timestamp"] = (
        analysis_cutoff
    )

    # ---------------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------------

    if rfm["customer_unique_id"].duplicated().any():
        raise ValueError(
            "RFM contains duplicate customer IDs"
        )

    if rfm["recency_days"].lt(0).any():
        raise ValueError(
            "Negative recency detected"
        )

    if rfm["frequency"].lt(1).any():
        raise ValueError(
            "Frequency must be at least 1"
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    output_path = GOLD_DIR / "customer_rfm.parquet"

    rfm.to_parquet(
        output_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("RFM MODEL VALIDATION: PASS")
    print("=" * 80)

    print(
        f"Customers: {len(rfm):,}"
    )

    print(
        f"Cutoff: {analysis_cutoff}"
    )

    print(
        f"Output: {output_path}"
    )

    print("\nRFM summary:")

    print(
        rfm[
            [
                "recency_days",
                "frequency",
                "monetary",
            ]
        ]
        .describe()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()