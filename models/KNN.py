from sklearn.neighbors import KNeighborsClassifier
import preprocess as prep

data = prep.data
#income1 = data.iloc["income"].values
#X = data.drop(['sex','race'], axis=1) #need to made a new data

# Before developing KNN I want to use sex and race as 4 group;
    #           Female and Non-White
    #           Female and Black
    #           Male and Non-White
    #           Male and Black

# Let's shape our data ;

#data['sex_race']=(data['sex']+data['race'])


# We have   0,0 => FN 2129
  #         0,1 => FW 8642
  #         1,0 => MN 2616
  #         1,1 => MW 19174

# def col_6(df):
#     df['sex_race'] = df[df['income'] == 0]['education-num'].values[0]
#     return df
#
# data.groupby(['sex','race']).apply(col_6)
#
# print(data.head())
#
#- sex_race = (data[data['income'] == 0].groupby(['sex','race'])['education_num'].first()).rename('sex_race')
# data = data.join(sex_race, on=['sex','race'])
# print(data.sex_race)
# sum_df = df.groupby(['year','month']).agg({'score': 'sum', 'num_attempts': 'sum'})


grouped_multiple = data.groupby(['sex', 'race']).agg({'income': ['mean'],'education-num':['mean']})
data = data.join(grouped_multiple, on=['sex','race'])
#print(data.tail)
#while education is high for the FW, income is low

#%%

from sklearn.model_selection import train_test_split

y =data['income']

X_train, X_test, y_train, y_test = train_test_split(data,y, test_size=0.5, random_state=42)

knn = KNeighborsClassifier(n_neighbors=4,metric='minkowski')
knn.fit(X_train,y_train.ravel())
result = knn.predict(X_test)
print(result)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,result)
print(cm)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, result)
print(accuracy)

# #%%
# test_data= prep.test_data
# X_test = test_data
# #X_test = X_train.reshape(len(X_test), -1)
# vectorizer = CountVectorizer()
# X_test = vectorizer.transform(X_test)
#
#
# res = knn.predict(X_test)
# X_test["income"] = res
# print(X_test)