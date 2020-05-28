import datetime as dt

import pandas as pd


# TODO: Unit tests


def compute_work_item_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with the ticket data and computes the start time, end time,  duration and the
    duration_in_hours.
    :param df: As described above
    :return: As described above
    """
    # We can't be sure that NAs were already renamed. Do it again just to be sure.
    df.from_phase.fillna('Start', inplace=True)
    df.to_phase.fillna('End', inplace=True)

    relevant_columns = ['work_item', 'timestamp']
    start_times = df[df.from_phase == 'Start'][relevant_columns]
    end_times = df[df.to_phase == 'End'][relevant_columns]
    times = pd.merge(start_times, end_times, on='work_item', how='left')
    times.rename(columns={'timestamp_x': 'start', 'timestamp_y': 'end'}, inplace=True)
    times['duration'] = times['end'] - times['start']
    times['duration_in_days'] = times['duration'].apply(lambda x: round(x.total_seconds() / (24*3600), 2))
    return times


def split_times(times: pd.DataFrame, sep_date_str: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits the times DataFrame into three different DataFrames
    - train: closed before the sep_date
    - test:  closed after the sep_date and started before the sep_date
    - closed: tickets that are closed
    - open:  tickets that arent't closed yet
    :param times: DataFrames containing the start and end times for each work items.
    :param sep_date_str: Separation date as string in the format DD.MM.YYYY
    :return: A tuple of (train, test, open) as defined above
    """
    open_times = times[pd.isnull(times.duration)]
    closed_times = times[~pd.isnull(times.duration)]
    sep_date = dt.datetime.strptime(sep_date_str, '%d.%m.%Y')
    train_times = closed_times[closed_times.end <= sep_date]
    test_times = closed_times[(closed_times.end > sep_date) & (closed_times.start <= sep_date)]
    return train_times, test_times, closed_times, open_times
