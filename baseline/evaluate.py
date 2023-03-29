import pandas as pd
import sys
from sklearn import metrics

preds=pd.read_csv(sys.argv[1],sep=',',header=None,names=['filename','category'])
preds['filename']=preds.filename.str.replace('.wav00000','')
preds.set_index('filename',inplace=True)

labels=pd.read_csv('../ESC-50-master/meta/esc50.csv')
labels.set_index('filename',inplace=True)
preds.rename(columns={'category':'pred'},inplace=True)

combined = preds.join(labels[['category']])

combined['category']= combined.category.map(lambda x:x.replace('_', ' '))

acc =  metrics.accuracy_score(combined.category,combined.pred)*100
balanced_acc =  metrics.balanced_accuracy_score(combined.category,combined.pred)*100

print(f'Total predictions: {len(combined)}')
print(f'Accuracy: {acc:.2f}%')
print(f'Balanced accuracy: {balanced_acc:.2f}%')
