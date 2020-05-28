from pathlib import Path
import pandas as pd
from graphviz import Digraph


def plot_open_and_closed_tickets(times: pd.DataFrame) -> None:
    """
    Plots the open and closed tickets per day
    :param times: Dataframe with the durations for each work item
    :return:
    """
    resample_period = 'D'  # We’re going to resample the dataframe per day
    open_per_day = times.resample(resample_period, on='start').work_item.count().rename('open_tickets_per_day')
    is_closed = times.end.notnull()
    closed_per_day = times.loc[is_closed] \
        .resample(resample_period, on='end') \
        .work_item.count() \
        .rename('closed_tickets_per_day')

    tickets_df = (pd.concat([open_per_day, closed_per_day], axis=1)
                  .fillna(0)
                  .astype(int)
                  .reset_index()
                  )

    tickets_df['open_tickets_total'] = tickets_df.open_tickets_per_day.cumsum()
    tickets_df['closed_tickets_total'] = tickets_df.closed_tickets_per_day.cumsum()
    tickets_df['wip_tickets_total'] = tickets_df.open_tickets_total - tickets_df.closed_tickets_total
    tickets_df.plot(x='start', y=['open_tickets_per_day', 'closed_tickets_per_day'],
                    figsize=(15, 10), subplots=True, alpha=0.5, sharey=True)
    return


def plot_process(process: list, name: str, path: str) -> Digraph:
    """
    Plots a graph from given process
    :param process: List of tuples with (from_phase, to_phase)
    :param name: name of the graph
    :param path: path where to save the file
    :return: graphviz graph object
    """
    g = Digraph(name=name, directory=Path(path))
    g.edges(process)
    g.attr(label=name)
    g.render()
    return g
