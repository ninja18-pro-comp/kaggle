import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import matplotlib
import seaborn as sns

train = pd.read_csv("train.csv",index_col="PassengerId")
test = pd.read_csv("test.csv",index_col="PassengerId")


preprocessed_train = pd.DataFrame()
preprocessed_train["Pclass"] = train["Pclass"]
preprocessed_train["Sex"] = train["Sex"].replace({'male': 0, 'female': 1})
preprocessed_train["Age"] = train["Age"].fillna(train["Age"].mean())
preprocessed_train["SibSp"] = train["SibSp"]
preprocessed_train["Fare"] = train["Fare"]
# preprocessed_train = pd.concat([preprocessed_train, pd.get_dummies(train["Embarked"])], axis=1)

x = preprocessed_train
y=train["Survived"]
sns.pairplot(pd.concat([y,x],axis=1)).savefig("pairplot.png")
# parameters = {
#     'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],
#     'gamma':[0.01,0.05,0.1,0.5,1,5,10,50,100],
# }
parameters = {
    'C':[10],
    'gamma':[0.01]
}
# 0.7654320987654321
# {'C': 10, 'gamma': 0.01}
svc = SVC(kernel='rbf')
clf = GridSearchCV(svc,parameters,cv=5)
clf.fit(x,y)
print(clf.best_score_)
print(clf.best_params_)
print(clf.cv_results_)
# joblib.dump(clf, 'clf.pkl') 
