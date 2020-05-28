import pytest
import pandas as pd
from src.data_processing.processmining import get_processes, process_frequency
from src.data_processing.functions import *
from src.data_processing.features import *


@pytest.fixture()
def short_data():
    data = pd.DataFrame({
        "work_item": ["WI_1", "WI_1", "WI_2", "WI_2", "WI_3"],
        "phase_col": ["Start-Analyze", "Analyze-End", "Start-Analyze", "Analyze-End", "Start-Analyze"],
        "duration_in_days": [1.5, 2.0, 0.89, 24.3, 0.75]
    })
    return data


@pytest.fixture()
def data():
    data = pd.DataFrame({
        'timestamp': [datetime.strptime("2017-01-01", "%Y-%m-%d"), datetime.strptime("2017-01-10", "%Y-%m-%d")],
        'work_item': ["WI_01", "WI_01"],
        'work_type': ["WT_01", "WT_01"],
        'work_priority': ["WP_02", "WP_02"],
        'domain': ["DOM_01", "DOM_01"],
        'platform': ["PL_07", "PL_07"],
        'components': ["COMP_99", "COMP_99"],
        'from_phase': ["Start", "Analyze"],
        'to_phase': ["Analyze", "Build"],
        'from_resource': ["ER_002", "ER_002"],
        'to_resource': ["ER_002", "ER_017"]
    })
    return data


@pytest.fixture()
def times(data):
    times = time_for_phase(data=data, process=False)
    times["receive_date"] = times["from_timestamp"].apply(lambda x: x.date())
    times["drop_date"] = times["to_timestamp"].apply(lambda x: x.date())
    return times


def test_get_processes(short_data):
    processes = get_processes(data=short_data, phase_col="phase_col")
    assert isinstance(processes, dict)
    assert "WI_1" in processes.keys()
    assert len(processes["WI_1"]) == 2


def test_process_frequency(short_data):
    processes = get_processes(data=short_data, phase_col="phase_col")
    procfreq = process_frequency(processes=processes)
    assert isinstance(procfreq, dict)
    assert procfreq[0]["process"] == processes["WI_1"]
    assert procfreq[0]["freq"] == 2


def test_resource_prep(short_data):
    processes = resource_prep(times=short_data, col="phase_col")
    assert len(processes) == 3


def test_set_end_date(data):
    end_date = "2020-12-24"
    with_end = set_end_date(data=data, end_date=end_date)
    assert len(with_end) == len(data) + 1
    assert with_end.loc[2, "timestamp"] == datetime.strptime(end_date, "%Y-%m-%d")


def test_time_for_phase(data):
    times = time_for_phase(data=data)
    assert set(["current_phase", "duration_in_days"]).issubset(set(times.columns))
    assert times[times["current_phase"] == "Analyze"].loc[0, "duration_in_days"] == 9


def test_x_experience(times):
    experience1 = x_experience(times=times,
                               resource="ER_017",
                               t=datetime.strptime("2017-01-10", "%Y-%m-%d").date())
    experience2 = x_experience(times=times,
                               resource="ER_002",
                               t=datetime.strptime("2017-01-19", "%Y-%m-%d").date())
    assert experience1 == 0
    assert experience2 == 9


def test_x_workload(times):
    workload1 = x_workload(times=times,
                           resource="ER_017",
                           t=datetime.strptime("2017-01-15", "%Y-%m-%d").date())
    workload2 = x_workload(times=times,
                           resource="ER_002",
                           t=datetime.strptime("2017-01-15", "%Y-%m-%d").date())
    assert workload1 == 1
    assert workload2 == 0


def test_calc_t_close(times):
    close_day = calc_t_close(times_closed=times, resource="ER_002")
    assert close_day == datetime.strptime("2017-01-10", "%Y-%m-%d").date()


def test_work_times(times):
    wt = work_times(times=times, time_col="duration_in_days")
    assert set(times["current_phase"].unique()) == set(wt.columns.drop("work_item"))
    assert wt.loc[0, "Analyze"] == 9


def test_process_length(times):
    le = process_length(times=times)
    assert le.loc[0, "process_length"] == 3


def test_is_open(data):
    io = is_open(data=data)
    assert io.loc[0, "is_open"] == 1


def test_is_outlier(times):
    out = is_outlier(times=times, time_col="duration_in_days")
    print(out.head())
    assert out.loc[0, "is_outlier"] == 1
