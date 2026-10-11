from pathlib import Path

import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    """Load the PCOS dataset."""
    return pd.read_excel(path)