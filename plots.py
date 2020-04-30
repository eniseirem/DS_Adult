import seaborn as sns
from collections import Counter
from matplotlib import pyplot as plt
import main as main
from math import pi

import pandas as pd

from bokeh.io import output_file, show
from bokeh.palettes import Category20c
from bokeh.plotting import figure
from bokeh.transform import cumsum
data = main.data
#I add clean data from main.py

target = data.values[:,0] #lets see education level distribution
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Education-level=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))

#%%
#plot this result.

#education_perc = sns.countplot(x='education-num', data=data)
#plt.show()

sns.barplot(x='education-num', y='education-num', data=data, estimator=lambda x: len(x) / len(data) * 100)
plt.show()

#sns.barplot(x='education-num', y='Values', data=data, estimator=lambda x: sum(x==0)*100.0/len(x))


#%% Education and income relation

print(data.groupby("education-num").mean() * 100)

pd.crosstab(data["education-num"],data.income).plot(kind='bar')
plt.title('Education and Income Relation')
plt.xlabel('Education')
plt.ylabel('Income')
plt.show()

#%% Gender and Income

pd.crosstab(data["sex"],data.income).apply(lambda r: r/r.sum() *100, axis=1).plot(kind='bar')
plt.title('Gender and Income Relation')
plt.xlabel('sex')
plt.ylabel('Income')
plt.show()

#as expected female income is much lower

#%%

print(data.groupby("income").mean())

#%%
# income_all = data.groupby(['education-num', 'income'])
# p = income_all.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
#
# print(p)
