from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm
from matplotlib import pyplot as plt
import seaborn as sns
import preprocess as prep


train_data = prep.data
X = train_data.drop(['income'], axis=1)
y = train_data['income']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
lr = LogisticRegression(random_state=42)
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