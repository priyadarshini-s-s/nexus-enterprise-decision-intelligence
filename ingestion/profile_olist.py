from pathlib import Path
import pandas as pd


BRONZE_DIR = Path("data/bronze/olist")


def inspect_csv(file_path: Path) -> pd.DataFrame:
    """Load a CSV for detailed profiling."""
    return pd.read_csv(file_path)


def profile_table(file_path: Path):
    """Generate structural and quality statistics for one table."""

    df = inspect_csv(file_path)

    print("\n" + "=" * 90)
    print(f"TABLE: {file_path.name}")
    print("=" * 90)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nDATA TYPES")
    print("-" * 40)
    print(df.dtypes)

    print("\nNULL VALUES")
    print("-" * 40)

    nulls = df.isnull().sum()
    null_pct = (nulls / len(df) * 100).round(2)

    null_report = pd.DataFrame({
        "null_count": nulls,
        "null_pct": null_pct
    })

    print(null_report[null_report["null_count"] > 0])

    print("\nDUPLICATE ROWS")
    print("-" * 40)
    print(df.duplicated().sum())

    print("\nUNIQUE VALUES")
    print("-" * 40)

    for column in df.columns:
        print(f"{column}: {df[column].nunique(dropna=True):,}")

    return df


def main():

    csv_files = sorted(BRONZE_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {BRONZE_DIR.resolve()}"
        )

    print("=" * 90)
    print("NEXUS — Olist Data Quality Profiling")
    print("=" * 90)

    for file_path in csv_files:
        profile_table(file_path)


if __name__ == "__main__":
    main()