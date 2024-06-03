import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def get_x_and_y_strat_test_data(strat_test_set):
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    return X_test, y_test


def add_additional_variables_to_X_test(X_test_prepared):
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )
    return X_test_prepared


def perform_necessary_processing(X_test, imputer):
    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )

    X_test_prepared = add_additional_variables_to_X_test(X_test_prepared)
    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))
    return X_test_prepared


def get_prediction(final_model, X_test_prepared, y_test):
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(f"RMSE is {final_rmse}")
    return final_predictions


def scoring_main_function(strat_test_set, imputer, final_model):
    X_test, y_test = get_x_and_y_strat_test_data(strat_test_set)
    X_test_prepared = perform_necessary_processing(X_test, imputer)
    final_predictions = get_prediction(final_model, X_test_prepared, y_test)
    print(final_predictions)
