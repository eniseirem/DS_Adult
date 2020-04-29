
import pandas as pd
import numpy as np

from sklearn import preprocessing

# %% First, lets take our data. Since there is no names for them, I look up the explanation from data informations put the names to labels
# some of them not seem clear to me, so i passed them and labeled as number (-n) however, they will not be part of my study so I will drop
#them in the end, so not a problem.
labels = [
"workclass",
"-1",
"education",
"education-num",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"-5",
"-3",
"-4",
"native-country",
"income",
]
#I'm adding these too avoid any type of na value.
missing_values = ["n/a", "na", "--", " ?","?"]

data = pd.read_csv('data/adult (1).data', names=labels, na_values=missing_values)
# Attribute Information:

# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# class: >50K, <=50K
# %%

data.drop(['workclass','-1','marital-status','occupation','relationship','-5','-3','-4','native-country'], axis=1, inplace=True)



# %% Now lets see, what our data look like.

#print(data.info())
#it says non-null so we're good. We don't need to do anything about missing values.
#I'll check the data for not make any mistake while working on it.
#print(data.values)


#%% Let's change our data to little bit to make it more analyzable. as we know we have
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# class: >50K, <=50K
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
#There is a " " in all data so, while changing the values I should be aware of them.

label_encoder = preprocessing.LabelEncoder()
data['income'] = label_encoder.fit_transform(data['income'])
data['sex'] = label_encoder.fit_transform(data['sex'])

# F=0 , M=1 ; -50k=0, +50K=1

#data['income'].replace([' <=50K',' >50K'],[0,1],inplace=True)
#data.sex.replace({' Female':0,' Male':1}, inplace=True)

#no reason to use different styles just want to see it, nothing different happened

#I changed the inputs as white and nonwhite.
data['race'] = [1 if x == ' White' else 0 for x in data['race']]

#Lets see our education inputs
print(data['education'].unique())
print(data['education-num'].unique())

#Education number only the numerical present of education. I'll combine the education
#names a little bit to reduce.
#We have various of inputs. Let's divide them.

# 5 Group ; No-education background : 0 ->  will be number 1 on table
#           Primary level : 1  -> 2,3,4,5...,8
#           Highschool level : 2 -> 9
#           Bachelor or college : 3 -> 10,11,12,13
#           Higher than Bachelor : 4 -> 13+

col_name='education-num'

conditions = [
    data[col_name].eq(1),
    data[col_name].isin([2,3,4,5,6,7,8]),
    data[col_name].eq(9),
    data[col_name].isin([10,11,12,13]),
    data[col_name].isin([14,15,16]),
]
result = [0,1,2,3,4]
data['education-num']=np.select(conditions,result)

#I do not need to keep education so;

data.drop(['education'], axis=1, inplace=True)

#%%


