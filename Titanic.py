"""
<a href="https://colab.research.google.com/github/Sbbarse787/Titanic-Data-analysis-and-ML-prediction/blob/master/Titanic.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("train.csv")
df.head(10)

df.shape

print(len(df))

# %%
df.index

# %%
df.columns

# %%
df.describe

# %%
df.isnull().sum()

# %%
df.dtypes

# %%
embark=pd.get_dummies(df["Embarked"],drop_first=True)
embark.head(10)                         # Creating a dummy variable for embark column

# %%
sex=pd.get_dummies(df['Sex'],drop_first=True)
sex.head(10)            #creating dummy variable for sex column

# %%
df=pd.concat([df,sex,embark],axis=1)
df.head()                               #Adding dummy variable of sex and embark in dataset

# %%
"""
Removing of unwanted columns...
"""

# %%
df.drop(['Sex','Embarked','PassengerId','Name','Ticket','Cabin','Fare'],axis=1,inplace=True)
df.head(5)                                                                                   #Droping unwanted columns

# %%
df.head()

# %%
print(len(df))

# %%
sns.countplot(x='Pclass',data=df)
#Number of PClass

# %%
sns.countplot(x='male',data=df)
#Shows the number of Male(1) and Female(0)

# %%

g = sns.catplot(x="Pclass", col="Survived",
                data=df, kind="count",
                height=4, aspect=.7);
                #Shows the Survival on the basis of Classes


# %%
"""
WRANGLING
"""

# %%
sns.boxplot(x="Pclass",y="Age",data=df)

# %%
"""
More aged people belongs to First class 
"""

# %%
df.dropna(inplace=True)

# %%
sns.heatmap(df.isnull(),yticklabels=False,cmap='Blues') #To check Our NAN Value is removed or not... and its removed

# %%
"""
Now We Dont have any Nan Values
"""

# %%
sns.boxplot(x="Pclass",y="Age",data=df)

# %%
"""
**NOW Lets do Training and Spliting Of Dataset**
"""

# %%
X=df.drop("Survived",axis=1)
y=df["Survived"]


# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# %%
"""
**LOGISTIC MODEL**
"""

# %%
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)

# %%
predictions=logmodel.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test,predictions)

# %%
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)

# %%
"""
So logistic is giving 76% precision...
"""

# %%
"""
Now Lets See How other classification model give precision to Our dataset
"""

# %%
"""
Lets start with **random forest**
"""

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# %%
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
pred_rfc=rfc.predict(X_test)

# %%
print(classification_report(y_test,pred_rfc))
print(confusion_matrix(y_test,pred_rfc))

# %%
 print(y_test[:10]) #checking first 10 datapoints which are predicted by Random forest with test data
 print(pred_rfc[:10])

# %%
"""
Sooo Rainforest is giving 81% accuracy quite Goood
"""

# %%
"""
Now lets check the other model
**SVM  Classifier**
"""

# %%
from sklearn import svm
from sklearn.svm import SVC

# %%
clf=svm.SVC()
clf.fit(X_train,y_train)
pred_clf=clf.predict(X_test)

# %%
print(classification_report(y_test,pred_clf))
print(confusion_matrix(y_test,pred_clf))

# %%
"""
SO Our SVM is giving a 68% Accuracy okkkk!
"""

# %%
"""
Now lets check another Model
"""

# %%
"""
**Decision Tree**
"""

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree, metrics, model_selection, preprocessing
from sklearn import tree
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtree.fit(X_train, y_train)

# %%
y_pred = dtree.predict(X_test)
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

# %%
print(y_test[:10],y_pred[:10])

# %%
"""
Decision Tree is giving as a accuracy of 81% quite good
"""

# %%
"""
Now lets check the accuracy With..
"""

# %%
"""
**Neural Network!!**
"""

# %%
from sklearn.neural_network import MLPClassifier

# %%
mlpc=MLPClassifier(hidden_layer_sizes=(9,9,9),max_iter=500)
mlpc.fit(X_train,y_train)
pred_mlpc=mlpc.predict(X_test)

# %%
print(classification_report(y_test,pred_mlpc))
print(confusion_matrix(y_test,pred_mlpc))

# %%
print(y_test[:10],pred_mlpc[:10])

# %%
"""
Ok so Neural Network is giving a Precision Of 81 % OKKK
"""

# %%
"""
So Lets Try the:
**Naive Bayes**
"""

# %%
from sklearn import naive_bayes, metrics, model_selection, preprocessing
gnb = naive_bayes.GaussianNB(priors=None)
gnb.fit(X_train_std, y_train)

# %%
sc = preprocessing.StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# %%
y_pred = gnb.predict(X_test_std)

# %%
# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

# %%
"""
So Naive Bayes is giving 74 % Accuracy 
"""
def add(d1,d2):
	return d1+d2

A=add(21,32)
	