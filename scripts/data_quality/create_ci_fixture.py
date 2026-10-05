from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


OUTPUT_PATH = Path(
    "tests/data_quality/fixtures/olist_orders"
)


def main() -> None:
    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.table(
        {
            "order_id": [
                "ci-order-001",
                "ci-order-002",
            ],
            "customer_id": [
                "ci-customer-001",
                "ci-customer-002",
            ],
            "order_status": [
                "delivered",
                "shipped",
            ],
            "order_purchase_timestamp": [
                datetime(2018, 1, 1, 10, 0, 0),
                datetime(2018, 1, 2, 11, 0, 0),
            ],
            "order_approved_at": [
                datetime(2018, 1, 1, 10, 30, 0),
                datetime(2018, 1, 2, 11, 30, 0),
            ],
            "order_delivered_carrier_date": [
                datetime(2018, 1, 2, 9, 0, 0),
                None,
            ],
            "order_delivered_customer_date": [
                datetime(2018, 1, 5, 12, 0, 0),
                None,
            ],
            "order_estimated_delivery_date": [
                datetime(2018, 1, 6, 0, 0, 0),
                datetime(2018, 1, 10, 0, 0, 0),
            ],
            "delivery_delay_days": [
                -1,
                None,
            ],
            "is_late_delivery": [
                0,
                0,
            ],
        }
    )

    output_file = OUTPUT_PATH / "part-00000.parquet"

    pq.write_table(
        table,
        output_file,
    )

    print(
        f"CI fixture written to: {OUTPUT_PATH}"
    )
    print(
        f"Rows: {table.num_rows}"
    )


if __name__ == "__main__":
    main()