import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rmsle(actuals: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """
    Computes the root mean square log error between the actuals and predictions.
    Raises and error if there are multiple predictions for a single work item, or if there are missing predictions
    :param actuals: A DataFrame with the columns 'work_item' and 'duration_in_days'
    :param predictions: A DataFrame with the columns 'work_item' and 'predictions'
    :return: RMSLE between actuals and predictions
    """
    assert len(actuals) == len(predictions)
    assert set(actuals.work_item.values) == set(predictions.work_item.values)
    actuals_values = actuals.duration_in_days.values
    predictions_values = predictions.predictions.values
    rmsle = np.sqrt(sum(((np.log(actuals_values + 1) - np.log(predictions_values + 1)) ** 2)) / len(actuals_values))
    return rmsle


def train_test_predictions(x_train: pd.DataFrame, x_test: pd.DataFrame, model,
                           scaler=None, val: int = None) -> pd.DataFrame:
    """
    Predicts values for train and test dataset
    :param x_train: Training dataset
    :param x_test: Testing dataset
    :param model: Trained model
    :param val: Which value you want if multiple values are predicted
    :return: 2 Dataframes, one with train one with test predictions
    """
    train_predictions = pd.DataFrame(x_train.index)
    test_predictions = pd.DataFrame(x_test.index)
    if scaler is None:
        x_train_vals = x_train.values
        x_test_vals = x_test.values
    elif scaler is not None:
        x_train_vals = scaler.fit_transform(x_train)
        x_test_vals = scaler.fit_transform(x_test)

    if val is None:
        train_predictions["predictions"] = model.predict(x_train_vals)
        test_predictions["predictions"] = model.predict(x_test_vals)
    elif val is not None:
        train_predictions["predictions"] = model.predict(x_train_vals)[:, val]
        test_predictions["predictions"] = model.predict(x_test_vals)[:, val]
    return train_predictions, test_predictions


def plot_train_test(train_predictions, y_train, test_predictions, y_test, col):
    """
    Plots the train and test predictions in two subplots
    :param train_predictions: Predictions for training set
    :param y_train: Target values for training set
    :param test_predictions: Predictions for test set
    :param y_test: Target values for test set
    :param col: Column name of the target value
    :return: None
    """
    assert col in y_train.columns, col + " not in y_train"
    assert col in y_test.columns, col + " not in y_test"

    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(15, 5), sharey=True, sharex=True)
    fig.suptitle('Actuals vs Predictions', fontsize=16)
    axs[0].scatter(x=y_train[col], y=train_predictions.predictions, alpha=0.1)
    axs[0].set_title('Train Data')
    axs[0].set_xlabel('to_predict')
    axs[0].set_ylabel('Predicted value')
    axs[1].scatter(x=y_test[col], y=test_predictions.predictions, alpha=0.1)
    axs[1].set_title('Test Data')
    axs[1].set_xlabel('to_predict')
