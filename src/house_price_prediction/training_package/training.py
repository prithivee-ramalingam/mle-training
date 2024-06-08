import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


def create_linear_regressor_model(housing_prepared, housing_labels):
    """
    Create and train a linear regression model.

    Parameters
    ----------
    housing_prepared : pandas.DataFrame
        The prepared housing data used for training the model.

    housing_labels : pandas.Series
        The target values (labels) corresponding to the housing data.

    Returns
    -------
    lin_reg : LinearRegression
        The trained linear regression model.
    """
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


def training_main_func(X_train, y_train):

    lin_reg = create_linear_regressor_model(X_train, y_train)
    lin_rmse = calculate_mean_squared_error(lin_reg, X_train, y_train)
    lin_mae = calculate_mean_absolute_error(lin_reg, X_train, y_train)

    print("RMSE for Linear Regression is ", lin_rmse)
    print("MAE for Linear Regression is ", lin_mae)

    tree_reg = create_decision_tree_model(X_train, y_train)
    tree_rmse = calculate_mean_squared_error(tree_reg, X_train, y_train)
    print(tree_rmse)

    create_random_forest_with_randomized_search_cv(X_train, y_train)

    grid_search = create_random_forest_with_grid_search(X_train, y_train)
    final_model = get_best_params_from_gs(grid_search, X_train)
    return final_model
