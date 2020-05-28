import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split

from data_access.load_file import load_agg_data
from modeling.evaluation import train_test_predictions, rmsle


def store_model(model, model_name: str):
    """
    Sores a model in a pickle file
    :param model: Trained model to store
    :param model_name: Name of the model
    :return: None
    """
    filepath = Path("../../models/") / (model_name + ".pkl")
    with open(filepath, "wb") as pkl:
        pickle.dump(model, pkl)


def split_data(times_closed: pd.DataFrame, feature_cols: list, test_size: int):
    """
    Splits data for training and testing
    :param times_closed: Data to split
    :param feature_cols: Columns to train model
    :param test_size: Size of the test set
    :return: x_train, x_test, y_train, y_test
    """
    x_data = times_closed[feature_cols]
    y_data = pd.DataFrame(times_closed["duration_in_days"])
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test


def model_predict(to_predict: pd.DataFrame, model, feature_cols: list = None):
    """
    Predicts values for a trained model
    :param to_predict: Dataframe with data you want to predict for
    :param model: Trained model
    :param feature_cols: Features from trained model
    :return: Predictions
    """
    to_predict.set_index("work_item", inplace=True)
    if feature_cols is not None:
        to_predict = to_predict[feature_cols]
    predictions = pd.DataFrame(to_predict.index)
    predictions["predictions"] = model.predict(to_predict.fillna(0))
    return predictions


def train_bayesianridge(feature_cols: list, test_size: float = 0.3, store: bool = True):
    """
    Trains a bayesian ridge model
    :param feature_cols: Features you want to train your model with
    :param test_size: Size of the test set
    :param store: Do you want to store your model in a pickle file?
    :return: Trained model
    """
    final, times_open, times_closed = load_agg_data()
    times_closed.set_index("work_item", inplace=True)

    x_train, x_test, y_train, y_test = split_data(times_closed, feature_cols, test_size)

    bay = BayesianRidge(n_iter=500)
    bay.fit(x_train, y_train.values.ravel())

    train_predictions, test_predictions = train_test_predictions(x_train, x_test, bay, val=None)

    train_rmsle = rmsle(y_train.reset_index(), train_predictions)
    test_rmsle = rmsle(y_test.reset_index(), test_predictions)
    print("Train error is %f" % train_rmsle)
    print("Test error is %f" % test_rmsle)

    if store is True:
        store_model(model=bay, model_name="BayesianRidge")

    return bay


if __name__ == "__main__":
    feature_cols = ["Accept_time", "Analyze_time", "Build_time", "Clarify_time", "Deploy_time", "Design_time",
                    "Package_time", "Test_time", "platcomp_cluster", "work_type", "work_priority"]
    _ = train_bayesianridge(feature_cols=feature_cols)

    final, times_open, times_closed = load_agg_data()
    # times_open = impute_times(final, times_open, times_closed, columns=feature_cols, imputation_method="forest")

    with open("../../models/BayesianRidge.pkl", "rb") as pkl:
        model = pickle.load(pkl)

    predictions = model_predict(to_predict=times_open, model=model, feature_cols=feature_cols)
    print(predictions.head())
