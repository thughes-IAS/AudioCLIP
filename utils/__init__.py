from . import datasets
from . import transforms

__all__ = [
    'datasets',
    'transforms'
]

def load_labels():
    import pandas as pd
    df = pd.read_csv('ESC-50-master/meta/esc50.csv')
    df['int_labels'] =df.filename.map(lambda x:int(x.split('-')[-1].split('.')[0]))
    out = pd.Series(dict(zip(df.int_labels,df.category))).sort_index().tolist()
    out = [x.replace('_', ' ') for x in out]
    return out

