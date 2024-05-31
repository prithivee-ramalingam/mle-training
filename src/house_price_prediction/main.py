from data_ingestion import data_ingestion_main_func
from scoring import scoring_main_function
from training import training_main_func


def main_function():
    housing = data_ingestion_main_func()
    strat_test_set, imputer, final_model = training_main_func(housing)
    scoring_main_function(strat_test_set, imputer, final_model)


main_function()
