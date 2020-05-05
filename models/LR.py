from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm
from matplotlib import pyplot as plt
import seaborn as sns
import preprocess as prep
import numpy as np


train_data = prep.data
X = train_data.drop(['income'], axis=1)
y = train_data['income']

#grouped_multiple = X.groupby(['sex', 'race']).agg({'education-num':['mean']})
#data = X.join(grouped_multiple, on=['sex','race'])
#print(grouped_multiple)
data = X
col_name='sex'
col_name2='race'
conditions = [
    data[col_name].eq(0) & data[col_name2].eq(0),
    data[col_name].eq(0) & data[col_name2].eq(1),
    data[col_name].eq(1) & data[col_name2].eq(0),
    data[col_name].eq(1) & data[col_name2].eq(1),


]
result = ["00", "01", "10", "11"]
data['s_r']=np.select(conditions,result)
print(data.tail())

X=data.drop(['sex','race'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# lr = LogisticRegression(random_state=0, class_weight="balanced") #0.703
#lr = LogisticRegression(random_state=42) #0.78
#lr = LogisticRegression(class_weight="balanced") #0.702
lr = LogisticRegression(random_state=4,class_weight="none") #0.702

lr.fit(X_train, y_train)

predictions = lr.predict(X_test)
score = round(accuracy_score(y_test, predictions), 3)
cm1 = cm(y_test, predictions)
sns.heatmap(cm1, annot=True, fmt=".0f")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Accuracy Score: {0}'.format(score), size = 15)
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions, target_names=['<=50K', '>50K']))

plt.figure(figsize=(8,8))
#%%
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc = roc_auc_score(y_test, lr.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label='LR (auc = %0.3f)' % roc_auc, color='navy')
plt.plot([0, 1], [0, 1],'r--')
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

#%%
from sklearn.feature_selection import RFE

rfe = RFE(lr, 10)
rfe = rfe.fit(X_train, y_train)
print(rfe.ranking_)
X_train[X_train.columns[rfe.ranking_==1].values].head()

#%%
import statsmodels.api as sm
from scipy import stats

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

logit_model=sm.Logit(y_train, X_train[X_train.columns[rfe.ranking_==1].values])
result=logit_model.fit()
predictions= result.predict(X_test[X_test.columns[rfe.ranking_==1].values])
print(classification_report(y_test, predictions.round(), target_names=['<=50K', '>50K']))
print(result.summary())

#%%

test_data= prep.test_data
X_test= test_data
#predictions = result.predict(X_test[X_test.columns[rfe.ranking_==1].values])
#print(result.summary)
predictions=lr.predict(X_test)
X_test["predict_income"] = predictions

#%%
import pandas as pd
data = X_test
pd.crosstab(data["sex"],data["predict_income"]).apply(lambda r: r/r.sum() *100, axis=1).plot(kind='bar')
plt.title('Sex and Income Relation')
plt.xlabel('sex')
plt.ylabel('Income')
plt.show()
class_weight="auto"

#%%
# from numpy import where
# counter = Counter(y)
# from matplotlib import pyplot
#
# for label, _ in counter.items():
# 	row_ix = where(y == label)[0]
# 	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
# pyplot.legend()
# pyplot.show()