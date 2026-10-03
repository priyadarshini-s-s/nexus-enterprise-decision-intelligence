from pathlib import Path

import numpy as np
import pandas as pd


GOLD_DIR = Path("data/gold/olist")


def main():
    print("=" * 80)
    print("NEXUS — RFM Clustering Dataset")
    print("=" * 80)

    rfm = pd.read_parquet(
        GOLD_DIR / "customer_rfm.parquet"
    )

    # Customers without observed payment data cannot be represented
    # meaningfully in monetary-based clustering.
    clustering = rfm[
        rfm["payment_observation_status"] == "observed"
    ].copy()

    print(f"\nTotal customers: {len(rfm):,}")
    print(
        "Customers with observed monetary value: "
        f"{len(clustering):,}"
    )
    print(
        "Excluded due to unavailable payment observation: "
        f"{len(rfm) - len(clustering):,}"
    )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    clustering["log_monetary"] = np.log1p(
        clustering["monetary"]
    )

    # Keep the raw variables as well as transformed monetary.
    # The clustering benchmark will decide the final feature space.
    output_columns = [
        "customer_unique_id",
        "recency_days",
        "frequency",
        "monetary",
        "log_monetary",
        "is_repeat_customer",
    ]

    clustering = clustering[output_columns]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if clustering["customer_unique_id"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs detected."
        )

    if clustering["recency_days"].lt(0).any():
        raise ValueError(
            "Negative recency detected."
        )

    if clustering["frequency"].lt(1).any():
        raise ValueError(
            "Invalid frequency detected."
        )

    if clustering["monetary"].lt(0).any():
        raise ValueError(
            "Negative monetary value detected."
        )

    if clustering["log_monetary"].isna().any():
        raise ValueError(
            "Missing transformed monetary values detected."
        )

    output_path = (
        GOLD_DIR /
        "rfm_clustering_dataset.parquet"
    )

    clustering.to_parquet(
        output_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("RFM CLUSTERING DATASET VALIDATION: PASS")
    print("=" * 80)

    print(f"Rows: {len(clustering):,}")
    print(f"Columns: {len(clustering.columns)}")
    print(f"Output: {output_path}")

    print("\nFeature summary:")
    print(
        clustering[
            [
                "recency_days",
                "frequency",
                "monetary",
                "log_monetary",
            ]
        ]
        .describe()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()