import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def transform_imputer(imputer, housing_num):
    """
    Transform the data using the provided imputer.

    Parameters
    ----------
    imputer : sklearn.impute._base.Imputer
        The imputer instance that has been fitted on the data.

    housing_num : pandas.DataFrame
        The numerical feature set to be transformed.

    Returns
    -------
    housing_tr : pandas.DataFrame
        The transformed numerical feature set.
    """
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
    return housing_tr


def get_household_related_information(housing):
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    return housing


def prepare_data(data_set, imputer):
    housing_num = data_set.drop("ocean_proximity", axis=1)
    housing_data = transform_imputer(imputer, housing_num)
    housing_data = get_household_related_information(housing_data)

    X_cat = data_set[["ocean_proximity"]]
    X = housing_data.join(pd.get_dummies(X_cat, drop_first=True))
    y = X[['median_house_value']]
    return X, y


def get_prediction(final_model, X_test_prepared, y_test):
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    return final_predictions, final_mse, final_rmse


def scoring_main_function(strat_test_set, imputer, final_model):
    X_test, y_test, imputer = prepare_data(strat_test_set, imputer)
    final_predictions, final_mse, final_rmse = get_prediction(
        final_model, X_test, y_test
    )
    print(final_predictions)
