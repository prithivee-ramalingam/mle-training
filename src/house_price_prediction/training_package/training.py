import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def perform_stratified_sampling(housing):
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    return strat_train_set, strat_test_set


def create_plots(housing):
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    without_ocean_proximity = housing.drop('ocean_proximity', axis=1)

    corr_matrix = without_ocean_proximity.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)


def get_household_related_information(housing):
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    return housing


def prepare_housing_data(housing, strat_train_set):
    housing = get_household_related_information(housing)
    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    return housing, housing_labels


def create_imputer(housing):
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    return imputer, housing_tr


def prepare_data_training(housing_tr, housing):
    housing_tr = get_household_related_information(housing_tr)
    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))
    return housing_cat, housing_prepared


def create_linear_regressor_model(housing_prepared, housing_labels):
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    return lin_reg


def create_decision_tree_model(housing_prepared, housing_labels):
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    return tree_reg


def create_random_forest_with_randomized_search_cv(housing_prepared, housing_labels):
    param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
    )

    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)


def create_random_forest_with_grid_search(housing_prepared, housing_labels):
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)
    return grid_search


def get_best_params_from_gs(grid_search, housing_prepared):
    grid_search.best_params_
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    print(sorted(zip(feature_importances, housing_prepared.columns), reverse=True))

    final_model = grid_search.best_estimator_
    return final_model


def calculate_mean_squared_error(model_object, housing_prepared, housing_labels):
    housing_predictions = model_object.predict(housing_prepared)
    mse = mean_squared_error(housing_labels, housing_predictions)
    rmse = np.sqrt(mse)
    return rmse


def calculate_mean_absolute_error(model_object, housing_prepared, housing_labels):
    housing_predictions = model_object.predict(housing_prepared)
    mae = mean_absolute_error(housing_labels, housing_predictions)
    return mae


def training_main_func(housing):
    strat_train_set, strat_test_set = perform_stratified_sampling(housing)
    housing = strat_train_set.copy()
    create_plots(housing)

    housing, housing_labels = prepare_housing_data(housing, strat_train_set)
    imputer, housing_tr = create_imputer(housing)
    housing_cat, housing_prepared = prepare_data_training(housing_tr, housing)

    lin_reg = create_linear_regressor_model(housing_prepared, housing_labels)
    lin_rmse = calculate_mean_squared_error(lin_reg, housing_prepared, housing_labels)
    lin_mae = calculate_mean_absolute_error(lin_reg, housing_prepared, housing_labels)

    print("RMSE for Linear Regression is ", lin_rmse)
    print("MAE for Linear Regression is ", lin_mae)

    tree_reg = create_decision_tree_model(housing_prepared, housing_labels)
    tree_rmse = calculate_mean_squared_error(tree_reg, housing_prepared, housing_labels)
    print(tree_rmse)

    create_random_forest_with_randomized_search_cv(housing_prepared, housing_labels)

    grid_search = create_random_forest_with_grid_search(
        housing_prepared, housing_labels
    )
    final_model = get_best_params_from_gs(grid_search, housing_prepared)
    return strat_test_set, imputer, final_model
