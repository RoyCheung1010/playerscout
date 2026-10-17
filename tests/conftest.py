"""Shared fixtures: the processed synthetic cohort, built once per test session."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.synthetic_generator import GeneratorConfig
from src.features.preprocessor import ScoutingPreprocessor


@pytest.fixture(scope="session")
def processed_players() -> pd.DataFrame:
    """Full synthetic DB after the loader + preprocessor, with ground truth kept."""
    df = DataLoader().load(generator_config=GeneratorConfig(include_ground_truth=True))
    return ScoutingPreprocessor().transform(df)


@pytest.fixture(scope="session")
def scaled_matrix(processed_players: pd.DataFrame) -> np.ndarray:
    X, _ = ScoutingPreprocessor().scale_features(processed_players, "standard")
    return X
