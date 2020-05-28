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


def my_rmsle(actuals: pd.DataFrame, predictions: pd.DataFrame, index_col: str, time_col: str) -> float:
    """
    Computes the root mean square log error between the actuals and predictions
    Raises and error if there are multiple predictions for a single work item, or if there are missing predictions
    :param actuals: Dataframe with training data
    :param predictions: Dataframe with predicted data
    :param index_col: work_items or other index to match on
    :param time_col: time you predicted
    :return: RMSLE between actuals and predictions
    """
    assert len(actuals) == len(predictions)
    assert set(actuals[index_col].values) == set(predictions[index_col].values)
    actuals_values = actuals[time_col].values
    predictions_values = predictions.predictions.values
    rmsle = np.sqrt(sum(((np.log(actuals_values + 1) - np.log(predictions_values + 1)) ** 2)) / len(actuals_values))
    return rmsle
