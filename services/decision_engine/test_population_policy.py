import pandas as pd

from services.decision_engine.population_policy import (
    prioritize_customers,
)


def test_five_percent_capacity():
    customers = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
                "C005",
                "C006",
                "C007",
                "C008",
                "C009",
                "C010",
            ],
            "repeat_purchase_probability": [
                0.01,
                0.09,
                0.03,
                0.20,
                0.05,
                0.01,
                0.15,
                0.02,
                0.07,
                0.04,
            ],
        }
    )

    result = prioritize_customers(
        customers,
        capacity_fraction=0.05,
    )

    prioritized = result[
        result["priority"] == "prioritize"
    ]

    assert len(prioritized) == 1
    assert prioritized.iloc[0]["customer_id"] == "C004"


def test_ten_percent_capacity():
    customers = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
                "C005",
                "C006",
                "C007",
                "C008",
                "C009",
                "C010",
            ],
            "repeat_purchase_probability": [
                0.01,
                0.09,
                0.03,
                0.20,
                0.05,
                0.01,
                0.15,
                0.02,
                0.07,
                0.04,
            ],
        }
    )

    result = prioritize_customers(
        customers,
        capacity_fraction=0.10,
    )

    prioritized = result[
        result["priority"] == "prioritize"
    ]

    assert len(prioritized) == 1
    assert prioritized.iloc[0]["customer_id"] == "C004"


def test_ranking_is_descending():
    customers = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
            ],
            "repeat_purchase_probability": [
                0.02,
                0.80,
                0.40,
            ],
        }
    )

    result = prioritize_customers(
        customers,
        capacity_fraction=0.50,
    )

    assert list(
        result["customer_id"]
    ) == [
        "C002",
        "C003",
        "C001",
    ]


def test_deterministic_tie_breaking():
    customers = pd.DataFrame(
        {
            "customer_id": [
                "C003",
                "C001",
                "C002",
            ],
            "repeat_purchase_probability": [
                0.50,
                0.50,
                0.50,
            ],
        }
    )

    result = prioritize_customers(
        customers,
        capacity_fraction=0.34,
    )

    assert list(
        result["customer_id"]
    ) == [
        "C001",
        "C002",
        "C003",
    ]

    assert result.iloc[0]["priority"] == "prioritize"


def test_invalid_capacity():
    customers = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "repeat_purchase_probability": [0.5],
        }
    )

    try:
        prioritize_customers(
            customers,
            capacity_fraction=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid capacity was accepted."
        )


def test_empty_population():
    customers = pd.DataFrame(
        columns=[
            "customer_id",
            "repeat_purchase_probability",
        ]
    )

    try:
        prioritize_customers(
            customers,
            capacity_fraction=0.10,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Empty population was accepted."
        )


def test_missing_required_column():
    customers = pd.DataFrame(
        {
            "customer_id": ["C001"],
        }
    )

    try:
        prioritize_customers(
            customers,
            capacity_fraction=0.10,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Missing required column was accepted."
        )


def test_invalid_probability():
    customers = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "repeat_purchase_probability": [1.5],
        }
    )

    try:
        prioritize_customers(
            customers,
            capacity_fraction=0.10,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid probability was accepted."
        )

def test_custom_customer_id_column():
    customers = pd.DataFrame(
        {
            "customer_unique_id": [
                "C003",
                "C001",
                "C002",
            ],
            "repeat_purchase_probability": [
                0.20,
                0.80,
                0.40,
            ],
        }
    )

    result = prioritize_customers(
        customers,
        capacity_fraction=0.34,
        customer_id_column="customer_unique_id",
    )

    assert list(
        result["customer_unique_id"]
    ) == [
        "C001",
        "C002",
        "C003",
    ]

    assert result.iloc[0]["priority"] == "prioritize"
print("=" * 80)
print("NEXUS — Decision Engine Policy Tests")
print("=" * 80)

test_five_percent_capacity()
print("5% capacity test: PASS")

test_ten_percent_capacity()
print("10% capacity test: PASS")

test_ranking_is_descending()
print("Ranking test: PASS")

test_deterministic_tie_breaking()
print("Tie-breaking test: PASS")

test_invalid_capacity()
print("Invalid capacity test: PASS")

test_empty_population()
print("Empty population test: PASS")

test_missing_required_column()
print("Missing-column test: PASS")

test_invalid_probability()
print("Invalid probability test: PASS")

print("\n" + "=" * 80)
print("ALL DECISION ENGINE TESTS PASSED")
print("=" * 80)