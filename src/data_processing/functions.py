from collections import OrderedDict, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd


def get_skipped_days(x: pd.Series) -> int:
    """
    Computes the number of release dates an item skipped
    :param x: Series with dates
    :return: Number of skipped dates
    """
    skipped_days = len([e for e in x["release_date"] if ((x["receive_date"] < e) & ~(x["drop_date"] <= e))])
    return skipped_days


def get_release_days(clustered: pd.DataFrame, total_times: pd.DataFrame) -> pd.DataFrame:
    """
    Gets the date and count of dropped items per cluster
    :param clustered: Dataframe with clustered items
    :param total_times: Dataframe with computed times
    :return: Dataframe with cluster and list of release dates
    """
    times_clustered = pd.merge(total_times, clustered).dropna()
    times_clustered.loc[:, "receive_date"] = times_clustered["start"].apply(lambda x: x.date())
    times_clustered.loc[:, "drop_date"] = times_clustered["end"].apply(lambda x: x.date())
    releases = pd.DataFrame(times_clustered.groupby("platcomp_cluster").apply(lambda x: x["drop_date"].value_counts()))
    releases.reset_index(inplace=True)
    releases.rename(columns={"level_1": "release_date", "drop_date": "wi_count"}, inplace=True)
    release_days = pd.DataFrame(
        releases.groupby("platcomp_cluster")["release_date"].apply(lambda x: x.values)).reset_index()
    return release_days


def calc_t_close(times_closed: pd.DataFrame, resource: str, percentage: float = 0.25):
    """
    Gets all days a given resource likes to drop a fixed percentage of his open items
    :param times_closed: Calculated times Dataframe from all closed items
    :param resource: Name of the resource
    :param percentage: Percentage when is a day a close day
    :return: List of days
    """
    res_df = times_closed[times_closed["current_resource"] == resource]
    #     We calculate the number of items dropped at a drop date
    t_close = pd.DataFrame(res_df["drop_date"].value_counts()).reset_index().rename(columns={"index": "drop_date",
                                                                                             "drop_date": "w_closed"})
    #     We calculate the work items that are open over a drop date
    res_df["w_open"] = res_df["drop_date"].apply(lambda t: res_df[(res_df["drop_date"].apply(lambda x: x >= t)) &
                                                                  (res_df["receive_date"].apply(
                                                                      lambda x: x <= t))].shape[0])
    #     Now we merge them together and calculate the percentage
    t_close = pd.merge(t_close, res_df[["drop_date", "w_open"]].drop_duplicates())
    t_close.loc[:, "percentage_closed"] = t_close["w_closed"] / t_close["w_open"].apply(lambda x: 1 if x == 0 else x)
    t_close.loc[:, "t_close"] = t_close["percentage_closed"].apply(lambda x: 1 if x >= percentage else 0)
    close_days = t_close[t_close["t_close"] == 1]["drop_date"].values
    return close_days


def get_school_holidays(year_range):
    """
    Gets the belgium school holiday dates as a dict
    :param year_range: Years to get holidays from
    :return: Dict with added holidays
    """
    schoolholidays = defaultdict(str)
    for year in year_range:
        if year == 2015:
            holidays = {"start": ["2014-12-22", "2015-02-16", "2015-04-06", "2015-07-01", "2015-11-02"],
                        "end": ["2015-01-04", "2015-02-22", "2015-04-19", "2015-08-31", "2015-11-08"]}
        elif year == 2016:
            holidays = {"start": ["2015-12-21", "2016-02-08", "2016-03-28", "2016-07-01", "2016-10-31"],
                        "end": ["2016-01-03", "2016-02-14", "2016-04-10", "2016-08-31", "2016-11-06"]}
        elif year == 2017:
            holidays = {"start": ["2016-12-26", "2017-02-27", "2017-04-03", "2017-07-01", "2017-10-30"],
                        "end": ["2017-01-08", "2017-03-05", "2017-04-16", "2017-08-31", "2017-11-05"]}
        elif year == 2018:
            holidays = {"start": ["2017-12-25", "2018-02-12", "2018-04-02", "2018-07-01", "2018-10-29"],
                        "end": ["2018-01-07", "2018-02-18", "2018-04-15", "2018-08-31", "2018-11-04"]}
        else:
            pass
        for start, end in zip(holidays["start"], holidays["end"]):
            daterange = pd.date_range(start=datetime.strptime(start, "%Y-%m-%d").date(),
                                      end=datetime.strptime(end, "%Y-%m-%d").date(), freq='D')
            for date in daterange:
                schoolholidays[date.date()] = "School holiday"
    return schoolholidays


def next_holidays(holiday: list, t: datetime.date):
    """
    Gets the minimum time distance between a date and a list
    :param holiday: List of holiday dates
    :param t: Date to compute distance for
    :return: Minimum number of days
    """
    minimum = min([round((x - t).total_seconds() / (24 * 3600), 2) for x in holiday if t <= x])
    return minimum


def employment_rate(times: pd.DataFrame, work_item: str, resource_col: str = "current_resource"):
    """
    Computes a value called employment rate for a work item (for more information see notebook 15_resource_measures)
    :param times: Dataframe with calculated times
    :param work_item: Name of the work item
    :param resource_col: Column name where resource are in
    :return: Employment rate, mean workload value and experience value
    """
    assert resource_col in times.columns, resource_col + " not in columns of dateframe!"

    wi = times[times["work_item"] == work_item]
    resources = list(wi[resource_col].values)
    res_counter = defaultdict(int)
    numerator = 0
    x_exp_sum = 0
    x_load_sum = 0
    denumerator = len(resources)

    for resource in resources:
        if resources.count(resource) > 1:
            res_counter[resource] += 1
            t = wi[wi[resource_col] == resource]["receive_date"].iloc[res_counter[resource] - 1]
        elif resources.count(resource) == 1:
            t = wi[wi[resource_col] == resource]["receive_date"].values[0]
        x_exp = x_experience(times, resource, t, resource_col)
        x_load = x_workload(times, resource, t, resource_col)
        x_exp_sum += x_exp
        x_load_sum += x_load
        numerator += (x_exp * x_load)

    x_emp = numerator / denumerator
    x_ex = x_exp_sum / denumerator
    x_l = x_load_sum / denumerator
    return x_emp, x_ex, x_l


def x_workload(times: pd.DataFrame, resource: str, t: datetime.date, col: str = "current_resource"):
    """
    Calculates an workload value for a resource at a given time (for more information see notebook 15_resource_measures)
    :param times: Dataframe with calculated times
    :param resource: Name of the resource
    :param t: Date to get experience from
    :param col: Column name where resource are in
    :return: Workload value
    """
    assert col in times.columns, col + " not in columns of dataframe"

    open_tickets = times[(times["receive_date"] <= t) &
                         (times["drop_date"] >= t)]
    er = open_tickets[open_tickets[col] == resource]

    try:
        x_load = (len(er)) / len(open_tickets)
    except ZeroDivisionError:
        x_load = 0

    return x_load


def x_experience(times: pd.DataFrame, resource: str, t: datetime.date, col: str = "current_resource"):
    """
    Calculates an experience value for a resource at a given time (for more information: notebook 15_resource_measures)
    :param times: Dataframe with calculated times
    :param resource: Name of the resource
    :param t: Date to get experience from
    :param col: Column name where resource are in
    :return: Experience value
    """
    assert col in times.columns, col + " not in columns of dataframe"

    closed_tickets = times[times["receive_date"] < t]
    er = closed_tickets[closed_tickets[col] == resource]

    try:
        date_diff = (t - er["receive_date"].min())
        working_time = round(date_diff.total_seconds() / (24 * 3600), 2)
    except TypeError:
        working_time = 1
    if pd.isna(working_time):
        working_time = 1

    try:
        x_exp = (working_time * len(er)) / len(closed_tickets)
    except ZeroDivisionError:
        x_exp = 0

    return x_exp


def show_open_tickets_on_date(df: pd.DataFrame, date: pd.Timestamp):
    """
    returns a data frame that includes information
    of open tickets on a given date
    df: pandas data frame object
    date: datetime
    :return: pandas data frame object
    """
    try:
        df1 = df[df.timestamp == date]
        Open_per_date = df1[df1.Closed == 0]
    except KeyError as e:
        return print('The Dataframe must include {0} as a column'.format(e))

    return Open_per_date


def show_closed_tickets_before_date(df: pd.DataFrame, date: pd.Timestamp):
    """
    returns a data frame that includes information
    of closed tickets before a given date
    df: pandas data frame object
    date: datetime
    :return: pandas data frame object
    """
    try:
        df1 = df[df.timestamp <= date]
        df_closed_tickets = df1[df1['Closed'] == 1]
    except KeyError as e:

        return print('The Dataframe must include {0} as a column'.format(e))
    return df_closed_tickets


def rmsle(Y_actual: np.array, Y_predict: np.array):
    """
    returns a float that represents the square
    root of the mean of the logarithmic
    differences between predicted and actual
    values. In general, a lower RMSLE is better
    than a higher one.
    Y_actual: Numpy array
    Y_predict: Numpy array
    :return: a float number
    """
    try:
        rmsle = np.sqrt(sum(((np.log(Y_predict + 1) - np.log(Y_actual + 1)) ** 2)) / len(Y_actual))
    except TypeError:
        return print('The inputs should be Numpy arrays')
    except ValueError:
        return print('The inputs should have equal length')

    return rmsle


def split_data(df: pd.DataFrame, target_name: str, size=0.6):
    """
    this function split the data frame
    with respect to the given size and
    provides four attributes
    x_train, y_train, x_test, y_test
    df: pandas data frame object
    target_name: string

    """
    try:
        nrow = int(len(df) * size)
        x_train = df.drop([target_name], axis=1)[:nrow]
        y_train = df[target_name][:nrow]
        x_test = df.drop([target_name], axis=1)[nrow:]
        y_test = df[target_name][nrow:]
        split_data.x_train = x_train
        split_data.y_train = y_train
        split_data.x_test = x_test
        split_data.y_test = y_test
    except (KeyError, TypeError):
        if KeyError:
            return print('second input must be a string')
        elif TypeError:
            return print('first input should be a dataframe')


def time_for_phase(data: pd.DataFrame, end_date: str = "2018-03-31",
                   relevant_columns: list = None, process=True) -> pd.DataFrame:
    """
    Calculates time for all proceeding phases
    :param data: Dataframe with raw data
    :param end_date: Set end date for items that are not closed yet in format "Year-Month-Day"
    :param relevant_columns: columns you want to have, do NOT put from_phase and to_phase here
    :param process: True if calculation based on process phases, False if calculation based on resource phases
    :return: Dataframe with calculated times
    """
    # We can't be sure that NAs were already renamed. Do it again just to be sure.
    data["from_phase"].fillna("Start", inplace=True)
    data["to_phase"].fillna("End", inplace=True)

    if end_date is not None:
        data = set_end_date(data=data, end_date=end_date)

    data.sort_values(by='timestamp', inplace=True)

    if relevant_columns is None:
        relevant_columns = ["work_item", "timestamp"]

    if process is True:
        from_phase = data[[*relevant_columns, "from_phase"]]
        to_phase = data[[*relevant_columns, "to_phase"]]
    elif process is False:
        from_phase = data[[*relevant_columns, "from_phase", "from_resource"]]
        to_phase = data[[*relevant_columns, "to_phase", "to_resource"]]

    # We add an index to every phase to join over it
    from_phase.loc[:, "process_index"] = from_phase.groupby("work_item").cumcount()
    to_phase.loc[:, "process_index"] = to_phase.groupby("work_item").cumcount()
    to_phase.loc[:, "process_index"] = to_phase["process_index"].apply(lambda x: x + 1)

    times = pd.merge(left=to_phase, right=from_phase, left_on=["work_item", "process_index"],
                     right_on=["work_item", "process_index"], how='left')

    # We rename and drop duplicated columns after the join
    new_names = {
        "to_phase": "current_phase",
        "to_resource": "current_resource",
        "timestamp_x": "from_timestamp",
        "timestamp_y": "to_timestamp"
    }
    times.rename(columns=new_names, inplace=True)
    times.drop(columns="from_phase", inplace=True)
    if process is False:
        times.drop(columns="from_resource", inplace=True)

    times.loc[:, "duration"] = times["to_timestamp"] - times["from_timestamp"]
    times.loc[:, "duration_in_days"] = times["duration"].apply(lambda x: round(x.total_seconds() / (24 * 3600), 2))
    # times.loc[:, "duration_in_hours"] = times["duration"].apply(lambda x: round((x.days*24) + (x.seconds/3600), 2))
    # times.loc[:, "duration_in_minutes"] = times["duration"].apply(lambda x: round((x.seconds / 60), 2))
    return times


def set_end_date(data: pd.DataFrame, end_date: str) -> pd.DataFrame:
    """
    Adds rows to the dataframe with a specific and date
    :param data: Raw data dataframe
    :param end_date: End date you want to set in format "Year-Month-Day"
    :return: Dataframe with added rows
    """
    if "End" in data["to_phase"].values:
        closed = data[data["to_phase"] == "End"]["work_item"].unique()
        opens = data[~data["work_item"].isin(closed)]["work_item"].unique()
    else:
        opens = data["work_item"].unique()
    with_end = data.copy()

    max_col = max(data.index)
    counter = 1
    for wi in opens:
        new_col = {'timestamp': datetime.strptime(end_date, "%Y-%m-%d"),
                   'work_item': wi,
                   'work_type': None,
                   'work_priority': None,
                   'domain': None,
                   'platform': None,
                   'components': None,
                   'from_phase': None,
                   'to_phase': "End",
                   'from_resource': None,
                   'to_resource': None}
        with_end.loc[max_col + counter] = new_col
        counter += 1
    return with_end


def resource_prep(times, col, stop=0):
    """
    Counts values of col and assigns to work_item
    :param times: Dataframe with calculated times
    :param col: Column name to count for
    :param stop: stop criteria
    :return: Dict of structure {work_item: col_values}
    """
    processes = OrderedDict()
    for wi in times["work_item"].unique():
        vals = times[times["work_item"] == wi][col].unique()
        processes[wi] = vals
        if stop > 0:
            stop -= 1
            if stop == 0:
                break
    return processes
