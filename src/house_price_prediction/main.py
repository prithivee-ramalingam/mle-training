from data_ingestion_package.data_ingestion import data_ingestion_main_func
from scoring_package.scoring import scoring_main_function
from training_package.training import training_main_func


def main_function():
    housing = data_ingestion_main_func()
    strat_test_set, imputer, final_model = training_main_func(housing)
    scoring_main_function(strat_test_set, imputer, final_model)


main_function()
