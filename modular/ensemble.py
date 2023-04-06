import pandas as pd
from glob import glob
import os
import numpy as np
from typing import cast


def ensemble(parent_dir: str,
             output_name: str):

    subs = [cast(pd.DataFrame,pd.read_csv(csv)) for csv in sorted(glob(os.path.join(parent_dir, '*.csv')))]

    n, _ = subs[0].shape
    n_csv = len(subs)
    wts = [1/n_csv]*n_csv
    labels = []
    names = subs[0].names

    for i in range(n):
        logits = np.sum([sub.iloc[i] * wts[i] for sub in subs])
        labels.append(np.argmax(logits))

    df = pd.DataFrame({
        'names': names,
        'class': labels
    })
    df.sort_values(by='names',
                   ascending=False,
                   inplace=True)
    df.to_csv(output_name, index=False)
