import pytest
from src.custom_filter_functions import (
    CustomGaussianKDE,
    CustomTSNE,
    CustomEccentricity,
    CustomPCA,
    CustomSVD,
)
from src.load_data import load_data


def test_custom_function_seeds():
    with pytest.raises(AttributeError) as ae:
        _ = CustomGaussianKDE(None, None)
        assert ae is not None
    with pytest.raises(KeyError) as ae:
        _ = CustomSVD(None, None, None)
        assert ae is not None
    with pytest.raises(AssertionError) as ae:
        _ = CustomPCA(None, None, None)
        assert ae is not None
    with pytest.raises(AssertionError) as ae:
        _ = CustomTSNE(None, None, None, None)
        assert ae is not None
    with pytest.raises(AssertionError) as ae:
        _ = CustomEccentricity(None, None, None, None)
        assert ae is not None


def test_path():
    with pytest.raises(FileNotFoundError) as e:
        _ = load_data("asdasfdfgsd")
        assert e is not None
