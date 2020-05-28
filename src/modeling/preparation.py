import pandas as pd
from missingpy import KNNImputer, MissForest


def impute_times(final, times_open, times_closed, columns, imputation_method="mean"):
    """
    Impute open work items times with different methods
    :param final: Complete preprocessed dataframe
    :param times_open: Dataframe of work items that are not closed
    :param times_closed: Dataframe of work items that are closed
    :param columns: Columns to impute
    :param imputation_method: Choose between 'mean', 'KNN', 'forest'
    :return: Dataframe of open work items with imputed values
    """
    if imputation_method == "mean":
        for col in columns:
            mean = times_closed[col].mean()
            mask = (times_open[col] == 0)
            times_open[col].mask(mask, mean, inplace=True)
    if imputation_method in ["KNN", "forest"]:
        if imputation_method == "KNN":
            imputer = KNNImputer(missing_values=0, col_max_missing=0.9)
        if imputation_method == "forest":
            imputer = MissForest(missing_values=0)
        for col in columns:
            try:
                val = imputer.fit_transform(pd.DataFrame(final[col]))[:, 0]
                other = pd.DataFrame(index=final.index, data=val, columns=[col])
                mask = (times_open[col] == 0)
                times_open.loc[mask, col] = other
            except ValueError:
                imputer = KNNImputer(missing_values=0, col_max_missing=0.99)
    return times_open
