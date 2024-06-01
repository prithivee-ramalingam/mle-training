import pandas as pd
from sklearn.metrics import mean_squared_error

from house_price_prediction import data_ingestion, scoring, training


def test_data_ingestion():
    data_ingestion.fetch_housing_data()
    housing = data_ingestion.load_housing_data()
    assert isinstance(housing, pd.DataFrame)


def test_training():
    housing = data_ingestion.load_housing_data()
    strat_train_set, strat_test_set = training.perform_stratified_sampling(housing)
    assert isinstance(strat_train_set, pd.DataFrame)
    assert isinstance(strat_test_set, pd.DataFrame)


def test_scoring():
    dict_data = {
        "median_house_value": [1000, 2000, 3000, 4000],
        "total_rooms": [12, 15, 18, 21],
        "total_bedrooms": [1500, 1300, 1100, 900],
    }
    strat_test_set = pd.DataFrame(dict_data)
    X_test, y_test = scoring.get_x_and_y_strat_test_data(strat_test_set)

    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
