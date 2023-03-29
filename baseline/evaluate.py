import pandas as pd
import sys
from sklearn import metrics

preds=pd.read_csv(sys.argv[1],sep=',',header=None,names=['filename','category'])
preds['filename']=preds.filename.str.replace('.wav00000','')
preds.set_index('filename',inplace=True)

labels=pd.read_csv('../ESC-50-master/meta/esc50.csv')
labels.set_index('filename',inplace=True)
preds.rename(columns={'category':'pred'},inplace=True)

labels['category']=labels.category.str.replace('_',' ')

combined = preds.join(labels[['category']])

combined['match']=(combined.pred==combined.category).astype(int)
accuracy1= 100*combined.match.value_counts(normalize=True)[1]
print(f'Total predictions: {len(combined)}')
print(f'Accuracy method 1: {accuracy1:.2f}%')

counts = combined.match.value_counts()
accuracy2=100*(counts[1]*50+counts[0]*48)/(50*counts.sum())
print(f'Accuracy method 2: {accuracy2:.2f}%')

acc =  metrics.accuracy_score(combined.category,combined.pred)
balanced_acc =  metrics.balanced_accuracy_score(combined.category,combined.pred)
print(f'Accuracy: {acc:.2f}')
print(f'Balanced accuracy: {balanced_acc:.2f}')
