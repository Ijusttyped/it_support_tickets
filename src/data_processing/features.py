from collections import defaultdict, OrderedDict

import holidays
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder

from src.data_access.load_file import dict_to_df
from src.data_preparation import compute_work_item_times
from src.data_processing.functions import resource_prep, employment_rate, next_holidays, get_school_holidays, \
    calc_t_close, get_release_days, set_end_date, get_skipped_days


def skipped_days(data, min_samples, times: pd.DataFrame = None) -> pd.DataFrame:
    """
    Computes the days an item skipped while an other item from the same cluster was dropped
    :param data: Raw data
    :param min_samples: Number of minimum samples in a cluster
    :param times: times dataframe if you want it phase-wise
    :return: Dataframe with new column skipped days
    """
    clustered = plat_comp_cluster(data, min_samples=min_samples)
    total_times = compute_work_item_times(data)
    release_days = get_release_days(clustered, total_times)
    return_cols = ["work_item", "skipped_days"]
    if times is None:
        total_times = compute_work_item_times(set_end_date(data, end_date="2018-03-31"))
    elif isinstance(times, pd.DataFrame):
        total_times = times.rename(columns={"from_timestamp": "start", "to_timestamp": "end"})
        return_cols.append("process_index")
    times_clustered = pd.merge(total_times, clustered, on="work_item").dropna()
    times_clustered.loc[:, "receive_date"] = times_clustered["start"].apply(lambda x: x.date())
    times_clustered.loc[:, "drop_date"] = times_clustered["end"].apply(lambda x: x.date())
    times_clustered = pd.merge(times_clustered, release_days, on="platcomp_cluster")
    times_clustered.loc[:, "skipped_days"] = times_clustered.apply(lambda x: get_skipped_days(x), axis=1)
    return times_clustered[return_cols]


def difficulty(times: pd.DataFrame, total_times: pd.DataFrame, index: bool = False) -> pd.DataFrame:
    """
    Calculates the difficulty value x_diff described in notebook 15_resource_measures
    :param times: Dataframe with calculated times per phase
    :param total_times: Dataframe with calculated per work_item
    :return: Dataframe with new column x_diff
    """
    return_df = times.copy()
    return_df.loc[:, "receive_date"] = return_df["from_timestamp"].apply(lambda x: x.date())
    return_df.loc[:, "drop_date"] = return_df["to_timestamp"].apply(lambda x: x.date())
    open_wis = total_times[pd.isnull(total_times["duration_in_days"])]["work_item"].values
    times_closed = return_df[~return_df.work_item.isin(open_wis)]
    times_closed.dropna(inplace=True)
    return_df.loc[:, "x_diff"] = 0
    for resource in return_df["current_resource"].unique():
        close_days = calc_t_close(times_closed, resource)
        res_df = return_df[return_df["current_resource"] == resource]
        for x, y in res_df.iterrows():
            difficulty = len([e for e in close_days if ((y["receive_date"] < e) & ~(y["drop_date"] <= e))])
            return_df.loc[x, "x_diff"] = difficulty
    cols = ["work_item", "x_diff"]
    if index is True:
        cols.append("process_index")
    elif index is False:
        return_df = pd.DataFrame(return_df.groupby("work_item")["x_diff"].sum()).reset_index()
    return return_df[cols]


def get_holidays(times: pd.DataFrame, timestamp_col: str = "from_timestamp", index: bool = False) -> pd.DataFrame:
    """
    Derives the information if the date is a belgium holiday and counts the days until the next holidays
    :param times: Dataframe with calculated times
    :param timestamp_col: Column name of the timestamp column
    :return: Dataframe with new columns is_holiday and days_to_nexthol
    """
    return_df = times.copy()
    year_range = range(2015, 2020)
    belhol = holidays.Belgium(years=list(year_range))
    schoolhol = get_school_holidays(year_range=year_range)
    return_df.loc[:, "is_holiday"] = return_df[timestamp_col].apply(lambda t: 1 if t in belhol else 0)
    return_df.loc[:, "days_to_nexthol"] = return_df[timestamp_col].apply(lambda t: next_holidays(belhol, t.date()))
    return_df.loc[:, "is_vacation"] = return_df[timestamp_col].apply(lambda t: 1 if t.date() in schoolhol.keys() else 0)
    return_df.loc[:, "days_to_nextvac"] = return_df[timestamp_col].apply(lambda t: next_holidays(schoolhol.keys(),
                                                                                                 t.date()))
    return_cols = ["work_item", "is_holiday", "days_to_nexthol", "is_vacation", "days_to_nextvac"]
    if index is True:
        return_cols.append("process_index")
    return return_df[return_cols]


def resource_measures(times: pd.DataFrame, resource_col: str = "current_resource") -> pd.DataFrame:
    # TODO: Replace loops with groupby
    """
    Calculates measures for the resources based on experience and workload
    :param times: Dataframe with calculated times
    :param resource_col: Column name where resources are in
    :return: New dataframe with employment_rate, experience_value, workload_value
    """
    times_copy = times.copy()
    times_copy.dropna(inplace=True)
    times_copy.loc[:, "receive_date"] = times_copy["from_timestamp"].apply(lambda x: x.date())
    times_copy.loc[:, "drop_date"] = times_copy["to_timestamp"].apply(lambda x: x.date())
    res_df = pd.DataFrame(index=times_copy.work_item.unique(), columns=["x_emp", "x_exp", "x_load"])
    counter = 0
    for wi in times_copy.work_item.unique():
        x_emp, x_exp, x_load = employment_rate(times=times_copy, work_item=wi, resource_col=resource_col)
        res_df.loc[wi, "x_emp"] = x_emp
        res_df.loc[wi, "x_exp"] = x_exp
        res_df.loc[wi, "x_load"] = x_load
        counter += 1
        if counter % 1000 == 0:
            done = len(res_df) - counter
            print(str(done) + " work items to do!")
    res_df.reset_index(inplace=True)
    res_df.rename(columns={"index": "work_item"}, inplace=True)
    return res_df


def resource_workload(times: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the workload of the current resource the time he picks up the work item
    :param times: Dataframe with calculated times
    :return: Dataframe with new column resource workload
    """
    times.dropna(inplace=True)
    times.loc[:, "receive_date"] = times["from_timestamp"].apply(lambda x: x.date())
    times.loc[:, "drop_date"] = times["to_timestamp"].apply(lambda x: x.date())
    daterange = pd.date_range(start=str(times["receive_date"].min()), end=str(times["drop_date"].max()), freq='D')
    workload = pd.DataFrame(index=daterange, columns=["working_resources", "total_resources"])
    for date in daterange:
        actual_working = times[(times["receive_date"] <= date.date()) & (times["drop_date"] >= date.date())]
        workload.loc[date, "working_resources"] = actual_working["current_resource"].nunique()
        monthly_workers = times[(times["receive_date"].apply(lambda x: x.year) == date.year) &
                                (times["receive_date"].apply(lambda x: x.month) <= date.month) &
                                (times["drop_date"].apply(lambda x: x.month) >= date.month)]
        workload.loc[date, "total_resources"] = monthly_workers["current_resource"].nunique()
    workload.loc[:, "resource_workload"] = workload["working_resources"] / workload["total_resources"]
    workload.reset_index(inplace=True)
    workload.loc[:, "receive_date"] = workload["index"].apply(lambda x: x.date())
    return_df = pd.merge(times, workload, on="receive_date", how="left")
    return return_df[["work_item", "resource_workload"]]


def resource_cluster(times: pd.DataFrame, resource_col: str = "current_resource", min_samples: int = 30):
    """
    Clusters resources for most likely to collaborate
    :param times: Dataframe with calculated times
    :param resource_col: Column of the resource
    :param min_samples: Minimum number of collaboration for clustering
    :return: Dataframe with new column resource_cluster
    """
    assert resource_col in times.columns, resource_col + " not in columns of dataframe"
    wi_res = resource_prep(times.dropna(), col=resource_col)
    res_df = dict_to_df(wi_res).T.rename(
        columns=dict(zip(range(8), [x + str(i) for x in ["resource_"] * 8 for i in range(8)])))
    # Encode the labels for transformation
    labelenc = LabelEncoder()
    for col in res_df.columns:
        res_df.loc[:, col + "_enc"] = labelenc.fit_transform(res_df[col].fillna("0"))
    enc_cols = [x for x in res_df.columns if x.endswith("enc") is True]
    dbscan = DBSCAN(min_samples=min_samples)
    cluster = dbscan.fit(res_df[enc_cols])
    res_df.loc[:, "resource_cluster"] = cluster.labels_
    res_df = res_df.reset_index().rename(columns={"index": "work_item"})
    return res_df[["work_item", "resource_cluster"]]


def plat_comp_cluster(data: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """
    Clusters domain, platform and components using DBSCAN
    :param data: Raw data table
    :param min_samples: Minimum number of samples to get in one cluster
    :return: Dataframe with new cluster column
    """
    to_cluster = data.sort_values(by=["work_item", "timestamp"]).drop_duplicates("work_item")[["work_item",
                                                                                               "domain",
                                                                                               "platform",
                                                                                               "components"]]
    to_cluster.fillna("0", inplace=True)
    labelenc = LabelEncoder()
    to_cluster.loc[:, "domain_encode"] = labelenc.fit_transform(to_cluster.domain)
    to_cluster.loc[:, "platform_encode"] = labelenc.fit_transform(to_cluster.platform)
    to_cluster.loc[:, "components_encode"] = labelenc.fit_transform(to_cluster.components)
    dbscan = DBSCAN(eps=0.5, min_samples=min_samples)
    clustering = dbscan.fit(to_cluster[["domain_encode", "platform_encode", "components_encode"]])
    to_cluster.loc[:, "platcomp_cluster"] = clustering.labels_
    return to_cluster[["work_item", "platcomp_cluster"]]


def has_loops(work_frequency: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a feature which is one if process phase appears multiple times, 0 if not
    :param work_frequency: Dataframe with calculated work frequencies for every phase
    :return: New dataframe with workitem and column "has_loops"
    """
    return_df = work_frequency.copy()
    return_df.set_index("work_item", inplace=True)
    mask = return_df.apply(lambda x: x > 1).sum(axis=1) > 0
    return_df.loc[:, "has_loops"] = 0.0
    return_df.loc[mask, "has_loops"] = 1.0
    return_df.reset_index(inplace=True)
    return return_df[["work_item", "has_loops"]]


def timestamp_information(times: pd.DataFrame, timestamp_col: str,
                          state: str = "start", index: bool = False) -> pd.DataFrame:
    """
    Derives time information about the given timestamp
    :param times: Dataframe with timestamp
    :param timestamp_col: Column name of the timestamp column
    :param state: If start or end timestamp, default is "start"
    :param index: process index to get info for every timestep
    :return: New dataframe with derived weekday, week, month and year
    """
    assert timestamp_col in times.columns, timestamp_col + " not in columns of dataframe"
    return_df = times.copy()
    return_df.loc[:, state + "_weekday"] = return_df[timestamp_col].apply(lambda x: x.weekday() + 1)
    return_df.loc[:, state + "_day"] = return_df[timestamp_col].apply(lambda x: x.day)
    return_df.loc[:, state + "_week"] = return_df[timestamp_col].apply(lambda x: x.week)
    return_df.loc[:, state + "_month"] = return_df[timestamp_col].apply(lambda x: x.month)
    return_df.loc[:, state + "_year"] = return_df[timestamp_col].apply(lambda x: x.year)
    return_cols = ["work_item", state + "_weekday", state + "_day", state + "_week", state + "_month", state + "_year"]
    if index is True:
        return_cols.append("process_index")
    return return_df[return_cols]


def is_outlier(times: pd.DataFrame, time_col: str, lim: int = 148) -> pd.DataFrame:
    """
    Adds the feature is_outlier based on duration limit
    :param times: Dataframe with calculated duration
    :param time_col: Column with duration time
    :param lim: Time limit, all above will be marked as outliers, default is 148 -> upper whisker
    :return: Dataframe with new column is_outlier
    """
    assert time_col in times.columns, time_col + " not in columns of dataframe"
    return_df = times.copy()
    return_df.loc[:, "is_outlier"] = 0
    outliers = return_df[return_df[time_col] > lim].index.values
    return_df.loc[outliers, "is_outlier"] = 1
    return_df = return_df.sort_values(by="is_outlier", ascending=False)
    return_df.drop_duplicates(subset=["work_item"], keep="first", inplace=True)
    return return_df[["work_item", "is_outlier"]].reset_index(drop=True)


def is_open(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the feature is_open with is one if the work_item is open and 0 if closed
    :param data: Raw data dataframe
    :return: Dataframe with new column is_open
    """
    assert "to_phase" in data.columns, "Dataframe must contain column to_phase"
    closed = data[data["to_phase"] == "End"]["work_item"].values
    opens = data[~data["work_item"].isin(closed)].index.values
    return_df = data.copy()
    return_df.loc[:, "is_open"] = 0
    return_df.loc[opens, "is_open"] = 1
    return return_df[["work_item", "is_open"]]


def process_length(times: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the process length of every work item
    :param times: Dataframe with time for phase data computed with functions.time_for_phase
    :return: Dataframe with work_item as index, process_length as value
    """
    assert "process_index" in times.columns, "Dataframe must contain column process_index"
    return_df = times.copy()
    return_df.sort_values(by="process_index", ascending=False, inplace=True)
    return_df.drop_duplicates(subset=["work_item"], keep='first', inplace=True)
    return_df.rename(columns={"process_index": "process_length"}, inplace=True)
    return_df.sort_values(by="work_item", inplace=True)
    return_df.reset_index(drop=True, inplace=True)
    return return_df[["work_item", "process_length"]]


def work_times(times: pd.DataFrame, phase_col: str = "current_phase", time_col: str = None,
               normalize: bool = False, stop=0) -> pd.DataFrame:
    """
    Get times for every process phase of a work item as a feature
    :param times: Dataframe with time for phase data computed with functions.time_for_phase
    :param phase_col: Name of the column with the values you want to have as column, default is 'current_phase'
    :param time_col: Name of the column you want to have the time from, if None the frequency is taken
    :param normalize: If phase times should by normalized by total time
    :param stop: Stop criteria to only iterate over specific number of rows
    :return: Dataframe with work_item as index, unique phase_col values as column, aggregated time_col values as value
    """
    assert phase_col in times.columns, phase_col + " not in dataframe"

    if time_col is None:
        times.loc[:, "frequency"] = 1
        time_col = "frequency"
    duration_sum = dict(times.groupby(["work_item", phase_col])[time_col].sum())
    all_phases = times[phase_col].unique()
    result = OrderedDict()

    for key, value in duration_sum.items():
        wi = key[0]
        phase = key[1]
        time = value
        if wi not in result.keys():
            result[wi] = defaultdict(dict, dict(zip(all_phases, [0] * len(all_phases))))
        result[wi][phase] = time
        if stop > 0:
            stop -= 1
            if stop == 0:
                break

    return_df = pd.DataFrame(result).T
    if time_col == "frequency":
        new_names = dict(zip(return_df.columns, [x + "_freq" for x in return_df.columns]))
        return_df.rename(columns=new_names, inplace=True)
    # Normalize by total duration
    if normalize is True:
        return_df = return_df.apply(lambda x: x / return_df.sum(axis=1))
        return_df.fillna(0, inplace=True)
    return_df.reset_index(inplace=True)
    return_df.rename(columns={"index": "work_item"}, inplace=True)
    return return_df
