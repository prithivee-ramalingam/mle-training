import os
import tarfile

import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Download data and extract it

    Parameters
    ----------
    housing_url : str
        The url where the data has to be donwloaded

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


def impute_data(X_train):
    housing = X_train.drop("median_house_value", axis=1)
    y = X_train["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")
    X_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(X_num)
    X = imputer.transform(X_num)

    X_prepared = pd.DataFrame(X, columns=X_num.columns, index=housing.index)
    return housing, y, X_prepared


def data_ingestion_main_func():
    fetch_housing_data()
    housing = load_housing_data()
    return housing
