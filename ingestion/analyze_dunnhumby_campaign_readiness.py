from pathlib import Path
from statistics import mean, stdev
import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "dunnhumby"

DB = duckdb.connect()

def csv_table(name, table):
    path = (RAW / name).as_posix().replace("'", "''")
    DB.execute(
        f"CREATE OR REPLACE TEMP TABLE {table} AS "
        f"SELECT * FROM read_csv_auto('{path}', header=true)"
    )

csv_table("campaign_desc.csv", "campaigns")
csv_table("campaign_table.csv", "assignments")
csv_table("transaction_data.csv", "transactions")
csv_table("hh_demographic.csv", "demographics")

DB.execute("""
    CREATE OR REPLACE TEMP TABLE households AS
    SELECT DISTINCT household_key FROM transactions
""")

def eligible_cohort(campaign_id):
    """Conservative overlap screen; assignment is not actual exposure."""
    row = DB.execute("""
        SELECT START_DAY, END_DAY
        FROM campaigns
        WHERE CAMPAIGN = ?
    """, [campaign_id]).fetchone()

    if row is None:
        raise ValueError(f"Campaign {campaign_id} not found")

    start_day, end_day = row

    DB.execute("DROP TABLE IF EXISTS cohort")
    DB.execute("""
        CREATE TEMP TABLE cohort AS
        WITH eligible_assignments AS (
            SELECT DISTINCT a.household_key
            FROM assignments a
            WHERE a.CAMPAIGN = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM assignments other
                  JOIN campaigns oc ON oc.CAMPAIGN = other.CAMPAIGN
                  WHERE other.household_key = a.household_key
                    AND other.CAMPAIGN <> ?
                    AND oc.START_DAY <= ?
                    AND oc.END_DAY >= ?
              )
        ),
        eligible_comparisons AS (
            SELECT h.household_key
            FROM households h
            WHERE NOT EXISTS (
                SELECT 1 FROM assignments a
                WHERE a.household_key = h.household_key
                  AND a.CAMPAIGN = ?
            )
            AND NOT EXISTS (
                SELECT 1
                FROM assignments other
                JOIN campaigns oc ON oc.CAMPAIGN = other.CAMPAIGN
                WHERE other.household_key = h.household_key
                  AND oc.START_DAY <= ?
                  AND oc.END_DAY >= ?
            )
        )
        SELECT household_key, 'assigned' AS group_name
        FROM eligible_assignments
        UNION ALL
        SELECT household_key, 'comparison' AS group_name
        FROM eligible_comparisons
    """, [
        campaign_id, campaign_id, end_day, start_day,
        campaign_id, end_day, start_day
    ])

    return start_day, end_day

def household_metrics(start_day, end_day):
    """Include eligible households with zero transactions in the window."""
    rows = DB.execute("""
        SELECT c.group_name, c.household_key,
               COALESCE(SUM(t.SALES_VALUE), 0.0) AS sales,
               COUNT(DISTINCT t.BASKET_ID) AS baskets,
               COUNT(DISTINCT t.DAY) AS active_days
        FROM cohort c
        LEFT JOIN transactions t
          ON t.household_key = c.household_key
         AND t.DAY BETWEEN ? AND ?
        GROUP BY c.group_name, c.household_key
        ORDER BY c.group_name, c.household_key
    """, [start_day, end_day]).fetchall()

    result = {"assigned": [], "comparison": []}
    for group, household, sales, baskets, active_days in rows:
        result[group].append({
            "household_key": household,
            "sales": float(sales),
            "baskets": int(baskets),
            "active_days": int(active_days),
        })
    return result

def smd(values_a, values_b, metric):
    a = [x[metric] for x in values_a]
    b = [x[metric] for x in values_b]

    if len(a) < 2 or len(b) < 2:
        return None

    # Match the reference report's equal-weighted group-variance convention.
    reference_variance = (stdev(a) ** 2 + stdev(b) ** 2) / 2

    if reference_variance <= 0:
        return None

    return abs(mean(a) - mean(b)) / reference_variance ** 0.5

print("\n===== BASELINE BALANCE: 28 DAYS BEFORE CAMPAIGN =====")
print("Campaign | Type | Assigned n | Comparison n | Sales SMD | Basket SMD | Active-day SMD")

for campaign_id in [8, 13, 18, 26, 29, 30]:
    start, end = eligible_cohort(campaign_id)
    metrics = household_metrics(start - 28, start - 1)
    a, b = metrics["assigned"], metrics["comparison"]

    kind = DB.execute(
        "SELECT DESCRIPTION FROM campaigns WHERE CAMPAIGN = ?",
        [campaign_id]
    ).fetchone()[0]

    values = [
        smd(a, b, "sales"),
        smd(a, b, "baskets"),
        smd(a, b, "active_days"),
    ]
    formatted = [
        "NA" if v is None else f"{v:.4f}" for v in values
    ]
    print(
        f"{campaign_id} | {kind} | {len(a)} | {len(b)} | "
        + " | ".join(formatted)
    )

print("\n===== PRE-CAMPAIGN WEEKLY SALES: CAMPAIGNS 26, 29, 30 =====")
print("Campaign | Week | Assigned mean sales | Comparison mean sales")

for campaign_id in [26, 29, 30]:
    start, end = eligible_cohort(campaign_id)

    # Exclude partial weeks and weeks overlapping campaign start.
    weeks = DB.execute("""
        SELECT WEEK_NO
        FROM transactions
        GROUP BY WEEK_NO
        HAVING MIN(DAY) < MAX(DAY)
           AND COUNT(DISTINCT DAY) = 7
           AND MAX(DAY) < ?
        ORDER BY WEEK_NO DESC
        LIMIT 8
    """, [start]).fetchall()
    weeks = sorted(row[0] for row in weeks)

    for week in weeks:
        rows = DB.execute("""
            SELECT c.group_name, c.household_key,
                   COALESCE(SUM(t.SALES_VALUE), 0.0) AS sales
            FROM cohort c
            LEFT JOIN transactions t
              ON t.household_key = c.household_key
             AND t.WEEK_NO = ?
             AND t.DAY < ?
            GROUP BY c.group_name, c.household_key
        """, [week, start]).fetchall()

        sales_by_group = {"assigned": [], "comparison": []}
        for group, household, sales in rows:
            sales_by_group[group].append(float(sales))

        a = mean(sales_by_group["assigned"]) if sales_by_group["assigned"] else 0
        b = mean(sales_by_group["comparison"]) if sales_by_group["comparison"] else 0
        print(f"{campaign_id} | {week} | {a:.4f} | {b:.4f}")

print("\n===== CAMPAIGN 30: BEFORE/DURING DESCRIPTIVE COMPARISON =====")
start, end = eligible_cohort(30)
before_start = start - (end - start + 1)
before_end = start - 1

before = household_metrics(before_start, before_end)
during = household_metrics(start, end)

for group in ["assigned", "comparison"]:
    print(f"\n{group.upper()} (n={len(before[group])})")
    for metric in ["sales", "baskets", "active_days"]:
        old = mean(x[metric] for x in before[group])
        new = mean(x[metric] for x in during[group])
        print(
            f"{metric}: before={old:.4f}, during={new:.4f}, "
            f"change={new - old:+.4f}"
        )

print("\n===== CAMPAIGN 30: DEMOGRAPHIC COVERAGE =====")
for group in ["assigned", "comparison"]:
    members = DB.execute(
        "SELECT household_key FROM cohort WHERE group_name = ?",
        [group]
    ).fetchall()
    ids = [row[0] for row in members]

    with_demo = DB.execute("""
        SELECT COUNT(DISTINCT c.household_key)
        FROM cohort c
        JOIN demographics d USING (household_key)
        WHERE c.group_name = ?
    """, [group]).fetchone()[0]

    total = len(ids)
    coverage = 100 * with_demo / total if total else 0
    print(
        f"{group}: eligible={total}, with_demographics={with_demo}, "
        f"coverage={coverage:.2f}%"
    )

print("\nDiagnostics completed. Results are observational, not causal.")
DB.close()
