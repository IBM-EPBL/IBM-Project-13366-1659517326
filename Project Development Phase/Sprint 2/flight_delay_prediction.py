#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import joblib
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier


# In[2]:


dataset = pd.read_csv('flightdata.csv')
dataset.head()


# In[3]:


dataset.columns


# In[4]:


dataset.dtypes


# In[5]:


# removing irrelevant data
dataset = dataset.drop(['Unnamed: 25', 'UNIQUE_CARRIER'], axis=1)


# In[6]:


# find exact value counts of all fields
cat_cols = dataset.select_dtypes(include=object).columns.tolist()
(pd.DataFrame(
    dataset[cat_cols]
    .melt(var_name='column', value_name='value')
    .value_counts())
.rename(columns={0: 'counts'})
.sort_values(by=['column', 'counts']))


# In[7]:


#plotting correlations (implication - almost linear relationship)
sb.jointplot(data=dataset, x="CRS_ARR_TIME", y="ARR_TIME")


# In[8]:


# creating a copy of the dataset for visualization

dataset_visualization = dataset.copy()


# In[9]:


dataset_visualization.dtypes


# In[10]:


list_str_obj_cols = dataset_visualization.columns[dataset_visualization.dtypes == "object"].tolist()
for str_obj_col in list_str_obj_cols:
    dataset_visualization[str_obj_col] = pd.to_numeric(dataset_visualization[str_obj_col], errors='coerce') 
dataset_visualization.dtypes


# In[11]:


correlations = dataset_visualization.corr()
sb.heatmap(correlations)


# In[12]:


# to identify the correlations between arrival delay and other fields
correlations['ARR_DEL15']


# In[13]:


# based on the correlation factors
refined_dataset = dataset.drop(['YEAR', 'QUARTER', 'DAY_OF_WEEK', 'TAIL_NUM', 'FL_NUM', 
                                'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'CRS_ELAPSED_TIME', 
                                'ACTUAL_ELAPSED_TIME', 'DISTANCE', 'ORIGIN',
                                'DEST', 'ARR_TIME', 'ARR_DELAY'], axis=1)
refined_dataset


# In[14]:


# check datatypes of columns
refined_dataset.dtypes


# In[15]:


# finding missing values
refined_dataset.isna().sum()


# In[16]:


# imputing missing values with mean
refined_dataset.dropna(inplace=True)
refined_dataset


# In[17]:


refined_dataset.shape


# In[18]:


# converting float values to categorical
columns = ['DEP_DEL15', 'CANCELLED', 'DIVERTED', 'ARR_DEL15']
for col in columns:
    refined_dataset[col] = refined_dataset[col].astype('int').astype('category')
refined_dataset.dtypes


# In[19]:


# converting float values to categorical
columns = ['DEP_TIME', 'DEP_DELAY']
for col in columns:
    refined_dataset[col] = refined_dataset[col].astype('int')
refined_dataset.dtypes


# In[20]:


# split dependent and independent variables
X = refined_dataset[['MONTH', 'DAY_OF_MONTH', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY',     
                  'DEP_DEL15', 'CRS_ARR_TIME', 'CANCELLED', 'DIVERTED']]     
Y = refined_dataset[['ARR_DEL15']]


# In[21]:


# splitting into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
y_train.value_counts()


# In[22]:


# decision tree classifier
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier = decision_tree_classifier.fit(X_train,y_train)
decision_tree_prediction = decision_tree_classifier.predict(X_test)
# performance metrics
print("Confusion matrix\n", confusion_matrix(decision_tree_prediction, y_test))
print("Classification report\n", classification_report(decision_tree_prediction, y_test))
print("Accuracy score\n", accuracy_score(decision_tree_prediction, y_test))


# In[23]:


# svc model
SVC_model = SVC()
SVC_model.fit(X_train, y_train)
SVC_prediction = SVC_model.predict(X_test)
# performance metrics
print("Confusion matrix\n", confusion_matrix(SVC_prediction, y_test))
print("Classification report\n", classification_report(SVC_prediction, y_test))
print("Accuracy score\n", accuracy_score(SVC_prediction, y_test))


# In[24]:


# knn model
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train, y_train)
KNN_prediction = KNN_model.predict(X_test)
# performance metrics
print("Confusion matrix\n", confusion_matrix(KNN_prediction, y_test))
print("Classification report\n", classification_report(KNN_prediction, y_test))
print("Accuracy score\n", accuracy_score(KNN_prediction, y_test))


# In[25]:


# gaussian naive bayes model
GNB_model = GaussianNB()
GNB_model.fit(X_train, y_train)
GNB_prediction = GNB_model.predict(X_test)
# performance metrics
print("Confusion matrix\n", confusion_matrix(GNB_prediction, y_test))
print("Classification report\n", classification_report(GNB_prediction, y_test))
print("Accuracy score\n", accuracy_score(GNB_prediction, y_test))


# In[26]:


# ensemble model of best 3 peforming model - gnb, knn, svc
ensemble = VotingClassifier(estimators=[('gnb', GNB_model), ('knn', KNN_model), ('svc', SVC_model)], voting='hard')
ensemble.fit(X_train, y_train)
ensemble_prediction = ensemble.predict(X_test)
# performance metrics
print("Confusion matrix\n", confusion_matrix(ensemble_prediction, y_test))
print("Classification report\n", classification_report(ensemble_prediction, y_test))
print("Accuracy score\n", accuracy_score(ensemble_prediction, y_test))


# In[27]:


joblib.dump(ensemble, 'flight.pkl')


# In[ ]:




