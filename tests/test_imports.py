import pytest


def test_import_data_ingestion():
    try:
        import house_price_prediction.data_ingestion
    except ImportError:
        pytest.fail("Failed to import data_ingestion")


def test_import_training():
    try:
        import house_price_prediction.training
    except ImportError:
        pytest.fail("Failed to import training")


def test_import_scoring():
    try:
        import house_price_prediction.scoring
    except ImportError:
        pytest.fail("Failed to import scoring")
