import pandas as pd
from src.data_access.load_file import *


def test_load_nested_data():
    filename = Path("./data/stg_feature_table.csv")
    data, times_open, times_closed = load_nested_data(filename)
    assert "work_item" in data.columns
    assert len(times_open.work_item.unique()) == 1042
    assert len(times_closed.columns) == 22


def test_load_agg_data():
    filename = Path("./data/agg_feature_table.csv")
    data, times_open, times_closed = load_nested_data(filename)
    assert "work_item" in data.columns
    assert len(times_open) == 1042
    assert len(times_closed.columns) == 49


def test_dict_to_df():
    testdict = {1: ["val1", "val2"], 2: ["val3"]}
    data = dict_to_df(testdict)
    assert isinstance(data, pd.DataFrame)


def test_add_phase():
    data = pd.DataFrame(data={"col1": 1, "col2": 2, "col3": 3}, index=range(1))
    new_data = add_phase(data=data, col_name="new_added", cols=["col1", "col2"])
    assert len(data.columns)+1 == len(new_data.columns)
