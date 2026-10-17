"""Tests for the 2-D tactical PCA projection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.config import CORE_FEATURES
from src.exceptions import InsufficientDataError
from src.models.pca import TacticalPCA


@pytest.fixture(scope="module")
def fitted(scaled_matrix: np.ndarray) -> TacticalPCA:
    return TacticalPCA().fit(scaled_matrix)


def test_output_shape(scaled_matrix: np.ndarray) -> None:
    coords = TacticalPCA().fit_transform(scaled_matrix)
    assert coords.shape == (scaled_matrix.shape[0], 2)
    assert np.isfinite(coords).all()


def test_fit_then_transform_equals_fit_transform(scaled_matrix: np.ndarray) -> None:
    np.testing.assert_allclose(
        TacticalPCA().fit(scaled_matrix).transform(scaled_matrix),
        TacticalPCA().fit_transform(scaled_matrix),
    )


def test_cumulative_variance_bounds(fitted: TacticalPCA) -> None:
    assert 0.0 <= fitted.cumulative_variance <= 1.0
    # Nine correlated football stats: two axes should keep a meaningful share.
    assert fitted.cumulative_variance > 0.4
    assert fitted.cumulative_variance == pytest.approx(fitted.explained_variance_ratio.sum())


def test_explained_variance_is_descending(fitted: TacticalPCA) -> None:
    ratio = fitted.explained_variance_ratio
    assert ratio.shape == (2,)
    assert ratio[0] >= ratio[1] > 0


def test_loadings_frame(fitted: TacticalPCA) -> None:
    loadings = fitted.get_loadings_df(list(CORE_FEATURES))
    assert list(loadings.index) == ["PC1", "PC2"]
    assert list(loadings.columns) == list(CORE_FEATURES)
    # Eigenvectors are unit length and orthogonal.
    np.testing.assert_allclose(np.linalg.norm(loadings.to_numpy(), axis=1), 1.0)
    assert loadings.loc["PC1"] @ loadings.loc["PC2"] == pytest.approx(0.0, abs=1e-9)


def test_pc1_separates_attack_from_defence(fitted: TacticalPCA) -> None:
    """The dominant axis should contrast attacking output with defensive work."""
    pc1 = fitted.get_loadings_df(list(CORE_FEATURES)).loc["PC1"]
    attack = pc1[["Goals_p90", "Shots_p90"]].mean()
    defence = pc1[["pAdj_Tackles_Won_p90", "pAdj_Interceptions_p90"]].mean()
    assert np.sign(attack) != np.sign(defence)


def test_loadings_name_count_must_match(fitted: TacticalPCA) -> None:
    with pytest.raises(ValueError):
        fitted.get_loadings_df(["only", "three", "names"])


def test_to_frame_columns(fitted: TacticalPCA, scaled_matrix: np.ndarray) -> None:
    frame = fitted.to_frame(fitted.transform(scaled_matrix[:5]), index=pd.Index(list("abcde")))
    assert list(frame.columns) == ["PC1", "PC2"]
    assert list(frame.index) == list("abcde")


def test_unfitted_access_raises() -> None:
    pca = TacticalPCA()
    assert not pca.is_fitted
    with pytest.raises(NotFittedError):
        pca.transform(np.zeros((3, 9)))
    with pytest.raises(NotFittedError):
        _ = pca.cumulative_variance


def test_too_few_samples_raises() -> None:
    with pytest.raises(InsufficientDataError):
        TacticalPCA().fit(np.zeros((1, 9)))


def test_zero_variance_input_is_handled() -> None:
    pca = TacticalPCA().fit(np.zeros((10, 9)))
    assert pca.cumulative_variance == 0.0
    assert np.isfinite(pca.transform(np.zeros((2, 9)))).all()


def test_bad_input_rejected(fitted: TacticalPCA) -> None:
    with pytest.raises(ValueError):
        TacticalPCA().fit(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        TacticalPCA().fit(np.array([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(ValueError):
        fitted.transform(np.zeros((3, 4)))


def test_deterministic(scaled_matrix: np.ndarray) -> None:
    np.testing.assert_array_equal(
        TacticalPCA().fit_transform(scaled_matrix), TacticalPCA().fit_transform(scaled_matrix)
    )
