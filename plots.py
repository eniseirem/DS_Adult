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


# %%
