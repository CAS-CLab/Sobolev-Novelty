from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd


@dataclass
class DataBundle:
    name: str
    data_settings: Any
    dataset_train: pd.DataFrame
    dataset_validation: pd.DataFrame
    dataset_test: pd.DataFrame
    target_columns: List[str]
    equation: Optional[str] = None
    operations_set: Optional[List[str]] = None
    data_dictionary: Optional[str] = None
    dataset_ood_test: Optional[pd.DataFrame] = None
