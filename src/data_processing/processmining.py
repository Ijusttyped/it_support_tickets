from collections import OrderedDict, defaultdict

import pandas as pd


def count_processes(data: pd.DataFrame, phase_col: str) -> defaultdict:
    """
    Counts all possible processes and stores in dict
    :param data: Dataframe with process data
    :param phase_col: Column name which you want to count
    :return: dict in style {index: {process:{i:tuple(from_phase, to_phase)}, freq: x, work_item: list(work_items}}
    """
    processes = get_processes(data=data, phase_col=phase_col)
    procfreq = process_frequency(processes)
    return procfreq


def process_frequency(processes: OrderedDict) -> defaultdict:
    """
    Counts processes frequency and stores in dict
    :param processes: dict with structure from data.exploration.processmining.get_processes
    :return: dict in style {index: {process:{i:tuple(from_phase, to_phase)}, freq: x, work_item: list(work_items}}
    """
    procfreq = defaultdict(dict)
    unique = []
    index = 0
    for key, value in processes.items():
        if str(value) not in unique:
            procfreq[index]["process"] = value
            procfreq[index]["freq"] = 1
            procfreq[index]["work_item"] = [key]
            unique.append(str(value))
            index += 1
        else:
            procfreq[unique.index(str(value))]["freq"] += 1
            procfreq[unique.index(str(value))]["work_item"].append(key)
    return procfreq


def get_processes(data: pd.DataFrame, phase_col: str) -> defaultdict:
    """
    Counts all possible processes and stores in dict
    :param data: Dataframe with process data
    :param phase_col: Column name which you want to count
    :return: dict in style {index: {process:{i:tuple(from_phase, to_phase)}, freq: x, work_item: list(work_items}}
    """
    data["process_index"] = data.groupby("work_item").cumcount()
    processes = dict(data.groupby("work_item").apply(lambda x: dict(zip(x["process_index"], x[phase_col]))))
    return processes
