from pathlib import Path
from typing import Any

import yaml


def load_contract(contract_path: str | Path) -> dict[str, Any]:
    """
    Load a NEXUS data contract from a YAML file.

    Parameters
    ----------
    contract_path:
        Path to the YAML contract.

    Returns
    -------
    dict
        Parsed contract.

    Raises
    ------
    FileNotFoundError
        If the contract file does not exist.
    ValueError
        If the YAML root is not a dictionary.
    """
    path = Path(contract_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Data contract not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        contract = yaml.safe_load(file)

    if not isinstance(contract, dict):
        raise ValueError(
            "Data contract must contain a YAML mapping/object at the root."
        )

    return contract