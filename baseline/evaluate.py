import pandas as pd

preds=pd.read_csv('../esc50.csv',sep=',',header=None,names=['filename','category'])
preds['filename']=preds.filename.str.replace('.wav00000','')
preds.set_index('filename',inplace=True)

labels=pd.read_csv('../ESC-50-master/meta/esc50.csv')
labels.set_index('filename',inplace=True)
