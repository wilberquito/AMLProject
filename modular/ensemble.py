"""
Script thought to ensemble predictions from differents models.
It requires an standard form of the predictions. All predictions
must have the same number of columns and the predictions ordered
by index. The indexes of all predictions must represent the name of
the image.
"""


import pandas as pd
from glob import glob
import os
import numpy as np


def ensemble(parent_dir: str,
             output_name: str,
             with_props=False):
    """
    Ensemble fun, all predictions are weighted the same
    """

    all_csv = []
    for csv in sorted(glob(os.path.join(parent_dir, '*.csv'))):
        all_csv.append(pd.read_csv(csv, index_col=0))

    # Computes how many all_csv has the first csv.
    # All csv has to have the same name of samples...
    n, _ = all_csv[0].shape
    # Computes how many different csv are
    n_csv = len(all_csv)
    # Weights for each csv
    wts = [1/n_csv]*n_csv
    # Get the name of all images
    names = list(all_csv[0].index)
    # Here I save the labels predicted by the ensemble
    labels = []
    # All probs
    probs = []

    # Per each sample, I compute the average weighted,
    # of the different predictions
    for i in range(n):
        all_preds = [csv.iloc[i].values * wts[j] for j, csv in enumerate(all_csv)]
        prob = np.sum(all_preds, axis=0)
        labels.append(np.argmax(prob, axis=0))
        probs.append(prob)

    df = pd.DataFrame({
        'name': names,
        'class': labels
    })
    df.sort_values(by='name',
                   ascending=True,
                   inplace=True)
    df.to_csv(output_name, index=False)

    if with_props:
        df = pd.DataFrame(probs)
        df.index = names
        df.to_csv('probs. ' + output_name, index=True)


if __name__ == '__main__':
    ensemble('./predictions', 'ensemble.csv')
