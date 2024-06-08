from data_ingestion_package.data_ingestion import data_ingestion_main_func
from scoring_package.scoring import scoring_main_function
from training_package.training import training_main_func


def main_function():

    X_train, y_train, imputer, strat_test_set = data_ingestion_main_func()
    final_model = training_main_func(X_train, y_train)
    scoring_main_function(strat_test_set, imputer, final_model)


main_function()
