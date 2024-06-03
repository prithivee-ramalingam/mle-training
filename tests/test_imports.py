import pytest


def test_import_data_ingestion():
    try:
        from house_price_prediction.data_ingestion_package import data_ingestion
    except ImportError:
        pytest.fail("Failed to import data_ingestion")


def test_import_training():
    try:
        from house_price_prediction.training_package import training
    except ImportError:
        pytest.fail("Failed to import training")


def test_import_scoring():
    try:
        from house_price_prediction.scoring_package import scoring
    except ImportError:
        pytest.fail("Failed to import scoring")
