import pandas as pd
from sklearn.linear_model import LinearRegression

from house_price_prediction.data_ingestion_package import data_ingestion
from house_price_prediction.scoring_package import scoring
from house_price_prediction.training_package import training


def test_data_ingestion():
    data_ingestion.fetch_housing_data()
    housing = data_ingestion.load_housing_data()
    assert isinstance(housing, pd.DataFrame)


def test_training():
    X_train, y_train, imputer, strat_test_set = (
        data_ingestion.data_ingestion_main_func()
    )
    lin_reg = training.create_linear_regressor_model(X_train, y_train)
    assert isinstance(lin_reg, LinearRegression)


def test_scoring():
    X_train, y_train, imputer, strat_test_set = (
        data_ingestion.data_ingestion_main_func()
    )
    housing_num = strat_test_set.drop("ocean_proximity", axis=1)
    housing_imputed = scoring.transform_imputer(imputer, housing_num)

    assert isinstance(housing_imputed, pd.DataFrame)
