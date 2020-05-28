import logging
import os
from pathlib import Path

import pandas as pd


def load_nested_data(filename=None):
    """
    Loads the preprocessed nested feature table
    :return: final, times_open, times_closed
    """
    if filename is None:
        filename = Path("../../data/stg_feature_table.csv")
    final = pd.read_csv(filename)
    is_open = final["is_open"] == 1
    times_open = final[is_open]
    times_closed = final[~is_open]
    return final, times_open, times_closed


def load_agg_data(filename=None):
    """
    Loads the preprocessed aggregated feature table
    :return: final, times_open, times_closed
    """
    if filename is None:
        filename = Path("../../data/agg_feature_table.csv")
    final = pd.read_csv(filename)
    is_open = final["is_open"] == 1
    times_open = final[is_open]
    times_closed = final[~is_open]
    return final, times_open, times_closed


def load_excel(file_name: str) -> pd.DataFrame:
    """
    Reads an excel file into a DataFrame. Looks for the file in the working directory and in the
    data/raw sub-folder. Throws FileNotFoundError

    :param file_name: Name of the excel file to load
    :return: A data frame containing the data in the excel file
    """
    data_folder = os.path.join('data')
    data_path = os.path.join(data_folder, file_name)
    raw_data_folder = os.path.join('data', 'raw')
    raw_data_path = os.path.join(raw_data_folder, file_name)
    processed_data_folder = os.path.join('data', 'processed')
    processed_data_path = os.path.join(processed_data_folder, file_name)
    project_paths = [file_name, data_path, raw_data_path, processed_data_path]
    notebook_paths = [os.path.join('..', p) for p in project_paths]
    search_paths = project_paths + notebook_paths
    for p in search_paths:
        try:
            excel_raw = pd.read_excel(p)
            logging.info('Found data source file at %s' % p)
            return excel_raw
        except FileNotFoundError:
            logging.info('Could not find data source file at %s' % p)
    msg = 'Could not find source data file ' + file_name
    raise FileNotFoundError(msg)


def load_table(filename: str) -> pd.DataFrame:
    """
    Loads csv table into pandas DataFrame
    :param filename: Path to file
    :return: pandas DataFrame
    """
    try:
        filename = Path(filename)
        data = pd.read_csv(filename, sep=";", parse_dates=['timestamp'])
        if set(["from_phase", "to_phase"]).issubset(set(data.columns)):
            data["from_phase"] = data["from_phase"].fillna("Start")
            data["to_phase"] = data["to_phase"].fillna("End")
        return data
    except FileNotFoundError:
        msg = "Colud not find source data file " + filename
        print(msg)
        logging.info(msg)


def dict_to_df(dicti: dict) -> pd.DataFrame:
    """
    Converts dictionary of different length into DataFrame
    :param dicti: Dict to convert
    :return: DataFrame from dict
    """
    return pd.DataFrame(dict([(key, pd.Series(value)) for key, value in dicti.items()]))


def add_phase(data: pd.DataFrame, col_name: str, cols: list, as_str=False) -> pd.DataFrame:
    """
    Adds a column to the DataFrame with tuple (cols[0], cols[1])
    :param data: Dataframe with raw data
    :param col_name: Name of new column
    :param cols: Names of existing columns to get tuple from
    :param as_str: New column as string object or not
    :return: Df with new column
    """
    assert len(cols) >= 2, "Column list must include at least 2 columns"
    new_data = data.copy()
    new_data[col_name] = tuple(zip(new_data[cols[0]], new_data[cols[1]]))
    if as_str is True:
        new_data["phase"] = new_data["phase"].apply(lambda x: str(x))
    return new_data
