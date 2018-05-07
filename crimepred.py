
# coding: utf-8

# In[153]:


import pandas as pd
import numpy as np
from sodapy import Socrata
import collections;
import re;
from time import time
from collections import defaultdict

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import PassiveAggressiveClassifier


# # Read Data Using API

# In[2]:


client = Socrata("data.sfgov.org", None)
data = client.get("cuks-n6tp", limit = 3000000)
data_df = pd.DataFrame.from_records(data)


# # Read data from CSV

# In[3]:


# import pandas as pd

# data_df = pd.read_csv("./data/crimedata.csv");


# In[4]:


# print(type(input))
# data_df = data_df.drop(data_df.columns[0], axis=1)


# In[5]:


print(type(data_df))
print(data_df.shape)


# In[6]:


print(data_df[:2])


# In[7]:


data_df.sort_values(by=['date', 'time'])
# data_df.to_csv('./data/crimedata.csv', sep= ",", encoding="utf8")


# In[8]:


print(data_df.shape)


# In[9]:


for col in data_df.columns:
    print(col)


# # Preprocessing

# In[10]:


def convert_date_to_day(dt):
    result = re.findall(r'\d{4}-(\d{2})-(\d{2})T00:00:00.000',dt)
    return result[0][1]
   
def convert_date_to_month(dt):
    result = re.findall(r'\d{4}-(\d{2})-(\d{2})T00:00:00.000',dt)
    return result[0][0]

def convert_time_to_hour(tm):
    result = re.findall(r'(\d{2}):\d{2}',tm)
    return result[0]

# def convert_date_to_year(dt):
#     result = re.findall(r'\d{4}-(\d{2})-(\d{2})T00:00:00.000',dt)
#     return result[0][0]


# In[11]:


data_df = data_df.dropna(how='any',axis=0)
print(data_df[:1])


# In[12]:


data_df['day'] = data_df.date.apply(lambda x: convert_date_to_day(x))
data_df['month'] = data_df.date.apply(lambda x: convert_date_to_month(x))
# data_df['year'] = data_df.date.apply(lambda x: convert_date_to_year(x))
data_df['hour'] = data_df.time.apply(lambda x: convert_time_to_hour(x))
data_df = data_df.sort_values(by=['date','time'])


# In[13]:


# df = data_df.drop(['incidntnum','pdid','resolution','x','y', 'date', 'time', 'descript'], axis =1)
df = data_df.drop(['incidntnum','pdid','resolution','date', 'time', 'descript', 'location'], axis =1)
# df = df.drop(df.columns[[0]], axis =1)

for col in df.columns:
    print(col)


# In[14]:


# df.sort_values(by=['date'])
df['category'] = df.category.apply(lambda x: x.lower())
df['dayofweek'] = df.dayofweek.apply(lambda x: x.lower())
df['address'] = df.address.apply(lambda x: x.lower())
df['pddistrict'] = df.pddistrict.apply(lambda x: x.lower())


# In[79]:


df['x'] = df.x.apply(lambda x: round(float(x),2))
df['y'] = df.y.apply(lambda y: round(float(y),2))


# # Unique Crime Categories

# In[16]:


uniqe_crime = {}
i = 0
for index,row in df.iterrows():
    if row['category'] not in uniqe_crime.keys():
        uniqe_crime[row['category']] = i
        i = i+1
    


# In[17]:


print(len(uniqe_crime))
print(uniqe_crime)


# In[18]:


# Define Index Crimes (More Serious)
index_crimes = ['ROBBERY', 'BURGLARY', 'LARCENY/THEFT', 'ASSAULT', 'ARSON', 'SEX OFFENSES, FORCIBLE', 'SECONDARY CODES',  'RECOVERED VEHICLE']
print(len(index_crimes))

# Define Non-Index Crimes (Less Serious)
non_index_crimes = ['OTHER OFFENSES', 'VEHICLE THEFT', 'NON-CRIMINAL', 'SUSPICIOUS OCC', 'FRAUD', 'FORGERY/COUNTERFEITING', 'WARRANTS', 'VANDALISM', 'MISSING PERSON', 'DISORDERLY CONDUCT', 'TRESPASS', 'WEAPON LAWS', 'DRUG/NARCOTIC', 'STOLEN PROPERTY', 'DRUNKENNESS', 'EMBEZZLEMENT', 'LOITERING', 'DRIVING UNDER THE INFLUENCE', 'PROSTITUTION', 'LIQUOR LAWS', 'EXTORTION', 'RUNAWAY', 'SUICIDE', 'BAD CHECKS', 'KIDNAPPING', 'FAMILY OFFENSES', 'BRIBERY', 'GAMBLING', 'SEX OFFENSES, NON FORCIBLE', 'PORNOGRAPHY/OBSCENE MAT', 'TREA']
len(non_index_crimes)


# # Find Severity

# In[19]:


severity = []
for index, row in df.iterrows():
    if row['category'].upper() in index_crimes:
        severity.append('INDEX')
#         print(row['category'], ": Index Crime (More Serious)")
    else:
        severity.append('NONINDEX')
#         print(row['category'], " : None Index Crime (More Serious)")


# In[20]:


print(len(severity))
print(df.shape)


# In[21]:


df['severity'] = severity


# In[22]:


print(df['category'][:10])
print(df['severity'][:10])


# In[23]:


print(df.shape)


# # Label Encoding to convert string to int

# In[24]:


lnc = LabelEncoder()


# In[25]:


def perform_label_encoding(temp_df):
    temp = lnc.fit_transform(temp_df.dayofweek.iloc[:].values)
    temp_df['dayofweek'] = temp
    temp = lnc.fit_transform(temp_df.address.iloc[:].values)
    temp_df['address'] = temp
    temp = lnc.fit_transform(temp_df.category.iloc[:].values)
    temp_df['category'] = temp
    return temp_df


# In[80]:


df = perform_label_encoding(df)
print(type(df))
print(df.shape)
print(df[:5])
# perform_label_encoding(sev_df)


# # Categorize all data by district in dictionary

# In[81]:


series = df.groupby('pddistrict').apply(list)
dictionary = {}


# In[82]:


series = df.groupby('pddistrict').apply(list)
dictionary = {}
city_target = df.get('severity').tolist()
unique_target=set(city_target)
print(len(unique_target))
print(len(city_target))


# In[83]:


for s in series.keys():
    print(str(s))
    mask = df['pddistrict'] == str(s)
#     print(df[mask])
    dictionary[str(s)] = df[mask]


# In[84]:


# for s in series.keys():
#     print(str(s))
#     mask = sev_df['pddistrict'] == str(s)
# #     print(df[mask])
#     dictionary[str(s)] = sev_df[mask]


# In[85]:


print(type(dictionary['central']))
print(dictionary.keys())


# # Method to fetch District data

# In[90]:


def fetchDistrictData(district):
    target = (dictionary.get(district))['severity']
    tempdf = (dictionary.get(district)).drop(['pddistrict', 'severity', 'category'], axis =1)
    return tempdf, target


# In[91]:


# def fetchDistrictData(district):
#     target = (dictionary.get(district))['category']
#     tempdf = (dictionary.get(district)).drop(['pddistrict', 'category','severity'], axis =1)
#     return tempdf, target


# In[92]:


district_bayview, district_bayview_target = fetchDistrictData('bayview')
print(len(district_bayview_target))
# print(district_bayview)


# In[93]:


# print(district_bayview.shape)
# print(district_bayview[:5])
print(len(set(district_bayview_target)))


# # Method to generate train and test data

# In[116]:


def fetchTrainTestData(district_df, district_target):
    X_train, X_test, y_train, y_test = train_test_split(district_df, district_target, test_size=0.20, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return X_train, X_test, y_train, y_test


# In[117]:


# X_train, X_test, y_train, y_test = fetchTrainTestData(mat_bayview, district_bayview_target)
# X_train, X_test, y_train, y_test = fetchTrainTestData(district_bayview, district_bayview_target)


# In[118]:


# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(type(X_train))
# print(type(X_test))
# print(type(y_train))
# print(type(y_test))


# In[119]:


# X_train_list = X_train.values.tolist()
# X_test_list = X_test.values.tolist()


# In[120]:


# print(X_train[:10])
# print(X_test[:10])


# # Method for classification

# In[121]:


def classify(clf, X_train, Y_train, X_test):
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, Y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    print(type(pred))
    return pred


# In[122]:


# from sklearn.preprocessing import StandardScaler
# X_train_std = StandardScaler().fit_transform(X_train.values.tolist())
# X_test_std = StandardScaler().fit_transform(X_train.values.tolist())


# In[123]:


# pred = classify(tree.DecisionTreeClassifier(), X_train, y_train, X_test)
# pred = classify(RandomForestClassifier(), X_train, y_train, X_test)
# pred = classify(KNeighborsClassifier(n_neighbors=3), X_train, y_train, X_test)
# pred = classify(AdaBoostClassifier(), X_train, y_train, X_test)


# In[174]:


def predictCrimeCategory(city, obj):
    district_df, district_target = fetchDistrictData(city)
    X_train, X_test, y_train, y_test = fetchTrainTestData(district_df, district_target)
    pred = classify(obj, X_train, y_train, X_test)
    score = f1_score(y_test, pred, average='macro')
    return pred, score


# In[175]:


def writeToFile(pred, path):
    print(path)
    with open(path,'w+') as f:
        for p in pred:
            f.write(str(p)+"\n")


# In[176]:


def classifyAllDistricts(clf, path = ""):
    dict_district_acc = {}
    dict_district_pred = {}
    dict_district_test = {}
    for key in dictionary.keys():
        pred, score, y_test = predictCrimeCategory(key, clf)
        dict_district_acc[key] = score
        dict_district_pred[key] = pred
        dict_district_test[key] = y_test
        writeToFile(pred, path+key+".dat")
    print(dict_district_acc)
    return dict_district_acc, dict_district_pred
#         


# In[139]:


decisiontree_parameters = {
    'splitter':['best','random'],
    'criterion' : ['gini','entropy'],
    'max_features' : ['auto', 'sqrt', 'log2'],
    }
neighbor_params = {
    'n_neighbors': [2, 3, 4], 
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}
randomforest_parameters = {
    'n_estimators':[10,20],
    'max_depth' : [None,50,70]
    }


# In[254]:


# # #Decision Tree
# clf = GridSearchCV(tree.DecisionTreeClassifier(), decisiontree_parameters)
knn_pred1, knn_score1 = predictCrimeCategory('tenderloin', KNeighborsClassifier(n_neighbors = 3))
print(pred, score)


# In[255]:


tree_pred1, tree_score1 = predictCrimeCategory('tenderloin', GridSearchCV(tree.DecisionTreeClassifier(), decisiontree_parameters))


# In[257]:


forest_pred1, forest_score1 = predictCrimeCategory('tenderloin', GridSearchCV(RandomForestClassifier(), randomforest_parameters))


# In[258]:


print(knn_score1, tree_score1)


# In[168]:


# knn_acc, knn_pred,  = classifyAllDistricts(KNeighborsClassifier(n_neighbors = 3), "./data/kneighbour/")


# In[167]:


# decisiontree_acc, decisiontree_pred = classifyAllDistricts(GridSearchCV(tree.DecisionTreeClassifier(), decisiontree_parameters), "./data/decisiontree/")


# In[166]:


# randomforest_acc, randomforest_pred = classifyAllDistricts(GridSearchCV(RandomForestClassifier(), randomforest_parameters), "./data/randomforest/")


# In[ ]:


# pred = classify(tree.DecisionTreeClassifier(), X_train, y_train, X_test)
# pred = classify(RandomForestClassifier(), X_train, y_train, X_test)
# pred = classify(KNeighborsClassifier(n_neighbors=3), X_train, y_train, X_test)
# pred = classify(AdaBoostClassifier(), X_train, y_train, X_test)


# In[159]:


final_pred = {}


# In[169]:


# for key in dictionary.keys():
# #     print(key)
#     pred_knn = kneighbor_pred.get(key)
#     pred_dtree = decisiontree_pred.get(key)
#     pred_forest = randomforest_pred.get(key)
#     ind = 0
#     non_ind = 0
#     list = []
#     length = len(pred_dtree)
#     print(length)
#     for i in range(length):
#         if pred_knn[i] is "INDEX":
#             ind+=1
#         else:
#             non_ind+=1
#         if pred_dtree[i] is "INDEX":
#             ind+=1
#         else:
#             non_ind+=1            
#         if pred_forest[i] is "INDEX":
#             ind+=1
#         else:
#             non_ind+=1
#         if ind>non_ind:
#             list.append("INDEX")
#         else:
#             list.append("NONINDEX")
#     print(len(list))
#     print(list[:5])
# #     final_pred[key] = list


# In[183]:


def generateAverageScore(kneighbor_pred, decisiontree_pred, randomforest_pred):
    list = []
    length = len(kneighbor_pred)
    print(length)
    for i in range(length):
        ind = 0
        non_ind = 0
        if kneighbor_pred[i] is "INDEX":
            ind+=1
        else:
            non_ind+=1
        if decisiontree_pred[i] is "INDEX":
            ind+=1
        else:
            non_ind+=1            
        if randomforest_pred[i] is "INDEX":
            ind+=1
        else:
            non_ind+=1
        if ind>non_ind:
            list.append("INDEX")
        else:
            list.append("NONINDEX")
    print(len(list))
    print(list[:5])
    return list
#     final_pred[key] = list


# In[200]:


def ensamble_classifiers(district, city = False):
    #Fetch district data
    if city is True:
        district_target = df['severity']
        district_df = df.drop(['pddistrict', 'severity', 'category'], axis =1)
    else:
        district_df, district_target = fetchDistrictData(district)
    X_train, X_test, y_train, y_test = fetchTrainTestData(district_df, district_target)
    #Classifiers
    kneighbor_pred  = classify(KNeighborsClassifier(n_neighbors = 3), X_train, y_train, X_test)
    score1 = f1_score(y_test, kneighbor_pred, average='macro')
    print("KNN: ", score1)
    decisiontree_pred = classify(GridSearchCV(tree.DecisionTreeClassifier(), decisiontree_parameters), 
                                X_train, y_train, X_test)
    score2 = f1_score(y_test, decisiontree_pred, average='macro')
    print("Decision Tree: ", score2)
    randomforest_pred = classify(GridSearchCV(RandomForestClassifier(), randomforest_parameters), 
                                 X_train, y_train, X_test)
    score3 = f1_score(y_test, randomforest_pred, average='macro')
    print("Random Forest: ", score3)
    
    list = generateAverageScore(kneighbor_pred, decisiontree_pred, randomforest_pred)
    score = f1_score(y_test, list, average='macro')
    print(district ,"Final F1 Score: " , score )
    return score


# In[261]:


def ensamble_classifiers_fetch_allscores(district, city = False):
    #Fetch district data
    if city is True:
        district_target = df['severity']
        district_df = df.drop(['pddistrict', 'severity', 'category'], axis =1)
    else:
        district_df, district_target = fetchDistrictData(district)
    X_train, X_test, y_train, y_test = fetchTrainTestData(district_df, district_target)
    #Classifiers
    kneighbor_pred  = classify(KNeighborsClassifier(n_neighbors = 3), X_train, y_train, X_test)
    score1 = f1_score(y_test, kneighbor_pred, average='macro')
    print("KNN: ", score1)
    decisiontree_pred = classify(GridSearchCV(tree.DecisionTreeClassifier(), decisiontree_parameters), 
                                X_train, y_train, X_test)
    score2 = f1_score(y_test, decisiontree_pred, average='macro')
    print("Decision Tree: ", score2)
    randomforest_pred = classify(GridSearchCV(RandomForestClassifier(), randomforest_parameters), 
                                 X_train, y_train, X_test)
    score3 = f1_score(y_test, randomforest_pred, average='macro')
    print("Random Forest: ", score3)
    
    list = generateAverageScore(kneighbor_pred, decisiontree_pred, randomforest_pred)
    score = f1_score(y_test, list, average='macro')
    print(district ,"Final F1 Score: " , score )
    scores = [score1, score2, score3, score]
    return scores


# In[192]:


ensamble_district_acc = {}


# In[193]:


for key in dictionary.keys():
    ensamble_district_acc[key] = ensamble_classifiers(key)


# In[202]:


city_acc = ensamble_classifiers(None, True)


# In[207]:


ensamble_district_acc['sanfrancisco'] = city_acc


# In[233]:


ensamble_district_acc.pop('sanfrancisco')


# In[239]:


# print(ensamble_district_acc)
print(ensamble_district_acc.keys())
# print(ensamble_district_acc.values())


# In[249]:


x_data = []
y_data = []


# In[250]:


for key in ensamble_district_acc.keys():
    x_data.append(key)
    y_data.append(ensamble_district_acc.get(key))
x_data.append("City")
y_data.append(city_acc)


# In[251]:


print(x_data)
print(y_data)


# In[252]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


# In[281]:


y_pos = np.arange(len(y_data))
 
plt.bar(y_pos, y_data, align='edge', alpha=0.767)
plt.xticks(y_pos, x_data, rotation=0)
plt.ylabel('F1 Score')
plt.title('Places')
plt.rcParams["figure.figsize"]=[16,8]
 
plt.show()


# In[262]:


scores = ensamble_classifiers_fetch_allscores('tenderloin')


# In[280]:


y_pos = np.arange(len(scores))
labels = ['KNN', 'Decision Tree', 'Random Forest', 'Ensemble']
plt.bar(y_pos, scores, align='edge', alpha=0.767)
plt.xticks(y_pos, labels, rotation=0)
plt.ylabel('F1 Score')
plt.title('district: tenderloin')
plt.rcParams["figure.figsize"]=[15,8] 
plt.show()


# In[301]:


randomforest_parameters1 = {
    'n_estimators':[10,20],
    'max_depth' : [None,50,70]
    }
randomforest_parameters2 = {
    'n_estimators':[10,20],
    'max_depth' : [None,5,10]
    }


# In[302]:


forest_pred2, forest_score2 = predictCrimeCategory('tenderloin', GridSearchCV(RandomForestClassifier(), 
                                                                              randomforest_parameters1))


# In[303]:


forest_pred3, forest_score3 = predictCrimeCategory('tenderloin', GridSearchCV(RandomForestClassifier(), 
                                                                              randomforest_parameters2))


# In[307]:


scores1 = [forest_score2, forest_score3]
y_pos = np.arange(len(scores1))
labels = ['Proper Parameter Tuning', 'Improper Parameter Tuning']
plt.bar(y_pos, scores1, align='edge', alpha=0.767)
plt.xticks(y_pos, labels, rotation=0)
plt.ylabel('F1 Score')
plt.title('district: tenderloin')
# plt.rcParams["figure.figsize"]=[8,5] 
plt.show()


# In[308]:


print(scores1)

