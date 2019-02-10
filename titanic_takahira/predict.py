import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import matplotlib
import seaborn as sns

train = pd.read_csv("test.csv",index_col="PassengerId")
preprocessed_train = pd.DataFrame()
preprocessed_train["Pclass"] = train["Pclass"]
preprocessed_train["Sex"] = train["Sex"].replace({'male': 0, 'female': 1})
preprocessed_train["Age"] = train["Age"].fillna(train["Age"].mean()).astype(int)
preprocessed_train["Age"].astype(int)
arr=[]
for v in train["Name"].values:
    arr.append(v.split(', ')[1].split(".")[0])
name = pd.DataFrame(arr)
name = name.where((name[0]=='Miss')|(name[0] =='Mr')|(name[0] =='Master')|(name[0] =='Mrs')|(name[0] =='Dr'),np.nan)
name.index = train.index
preprocessed_train = pd.concat([preprocessed_train, pd.get_dummies(name)], axis=1)
preprocessed_train["SibSp"] = train["SibSp"]
preprocessed_train["Fare"] = train["Fare"].fillna(train["Fare"].mean())
preprocessed_train = pd.concat([preprocessed_train, pd.get_dummies(train["Embarked"])], axis=1)
x = preprocessed_train
x.to_csv("preprocessed_test")
print(x)
clf = joblib.load('clf.pkl')
result = clf.predict(x)
y_hat = pd.DataFrame(result)
y_hat = y_hat.rename({0:"Survived"},axis=1)
y_hat.index = x.index
y_hat.to_csv("y_hat.csv")
print(y_hat)