import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Download data and extract it

    Parameters
    ----------
    housing_url : str
        The url where the data has to be downloaded

    housing_path : str
        The path to store data

    Returns
    -------
    None
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """
    Load data from the given path

    Parameters
    ----------
    housing_path : str
        The local path from which the data has to be read.

    Returns
    -------
    pandas.DataFrame
        The loaded data as a pandas DataFrame.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def perform_stratified_sampling_based_on_income_category(housing):
    """
    Performs stratified split on housing dataframe

    Parameters
    ----------
    housing : pandas.DataFrame
        Raw Housing Dataframe

    Returns
    -------
    strat_train_set : pandas.DataFrame
        stratified train set after undergoing split
    strat_test_set : pandas.DataFrame
        stratified test set after undergoing split
    """

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    return strat_train_set, strat_test_set


def fit_imputer(housing_num):
    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing_num)
    return imputer


def transform_imputer(imputer, housing_num):
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
    return housing_tr


def get_household_related_information(housing):
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    return housing


def prepare_data(data_set, manner, imputer=None):
    housing_num = data_set.drop("ocean_proximity", axis=1)
    if manner == 'train':
        imputer = fit_imputer((housing_num))
    housing_data = transform_imputer(imputer, housing_num)
    housing_data = get_household_related_information(housing_data)

    X_cat = data_set[["ocean_proximity"]]
    X = housing_data.join(pd.get_dummies(X_cat, drop_first=True))
    y = X[['median_house_value']]
    if manner == 'train':
        return X, y, imputer
    return X, y, None


def data_ingestion_main_func():
    fetch_housing_data()
    housing = load_housing_data()
    strat_train_set, strat_test_set = (
        perform_stratified_sampling_based_on_income_category(housing)
    )
    X_train, y_train, imputer = prepare_data(strat_train_set, 'train')
    return X_train, y_train, imputer, strat_test_set
