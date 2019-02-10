import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import matplotlib
import seaborn as sns
import chainer
import xgboost
from sklearn.semi_supervised import LabelSpreading

def preprocess_1(train):
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
    if "Survived" in train.columns:
        x["Survived"] = train["Survived"]
    return x

def preprocess_semi(train,unlabeled):
    unlabeled["Survived"] = -1
    train = pd.concat([train,unlabeled])
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
    x["Survived"] = train["Survived"]
    xy = x
    labeled_xy = xy[xy["Survived"]!=-1]
    unlabeled_xy = xy[xy["Survived"]==-1]
    labeled_x = labeled_xy.loc[:,labeled_xy.columns != "Survived"]
    labeled_y = labeled_xy.loc[:,labeled_xy.columns == "Survived"]
    unlabeled_x = unlabeled_xy.loc[:,unlabeled_xy.columns!="Survived"]
    unlabeled_y = unlabeled_xy.loc[:,unlabeled_xy.columns=="Survived"]

    return labeled_x,labeled_y,unlabeled_x,unlabeled_y

def svm(x,y):
    parameters = {
        'C': [1, 10, 100, 1000],
        'gamma': [0.001, 0.0001], 
        'kernel': ['rbf']
    }
    svm = SVC()
    clf = GridSearchCV(svm,parameters,cv=5)
    clf.fit(x,y)
    print(clf.best_score_)
    # print(clf.best_params_)
    # print(clf.cv_results_)
    joblib.dump(clf.best_estimator_, 'svm.pkl')

def gradboost(x,y):
    parameters = {
        'max_depth':[x for x in range(3,10,2)],
        'min_child_weight':[x for x in range(1,6,2)]  
    }
    gb = xgboost.XGBClassifier()
    clf = GridSearchCV(gb,parameters,cv=5)
    clf.fit(x,y)
    print(clf.best_score_)
    print(clf.best_params_)
    joblib.dump(clf.best_estimator_, 'gb.pkl')

def labelspreading(x,y,unlabeled_x,unlabeled_y):
    parameters = {
        "gamma":[0.01,0.05,0.1,0.5,1,5,10],
        "max_iter":[1000]
    }
    x["Survived"] = y
    xy = x
    unlabeled_x["Survived"] = unlabeled_y
    unlabeled_xy = unlabeled_x
    print(unlabeled_xy)
    for r in range(10+1):
        tmp = unlabeled_xy.sample(frac=float(r)/10.0)
        xy2 = pd.concat([xy,tmp])
        x = xy2.loc[:, xy2.columns != "Survived"]
        y = xy2["Survived"]
        ls= LabelSpreading()
        clf = GridSearchCV(ls,parameters,cv=5)
        clf.fit(x,y)
        print(clf.best_score_)
    # print(clf.best_params_)
    # print(clf.cv_results_)
    joblib.dump(clf.best_estimator_, 'lsclf.pkl')

def linerdnn():
    pass
    # model = dnn.dnn()
    # if args.gpu >= 0:
    #     # Make a specified GPU current
    #     chainer.backends.cuda.get_device_from_id(args.gpu).use()
    #     model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    # optimizer = chainer.optimizers.Adam()
    # optimizer.setup(model)

def predict(modelpath,x):
    clf = joblib.load(modelpath)
    result = clf.predict(x)
    y_hat = pd.DataFrame(result)
    y_hat = y_hat.rename({0:"Survived"},axis=1)
    y_hat.index = x.index
    return y_hat

if __name__ == '__main__':
    train = pd.read_csv("train.csv",index_col="PassengerId")
    test = pd.read_csv("test.csv",index_col="PassengerId")
    xy = preprocess_1(train)
    x = xy.loc[:,xy.columns!="Survived"]
    y = xy.loc[:,xy.columns=="Survived"]
    print(x)
    print(y)
    # x,y,x2,y2= preprocess_semi(train,test)
    # svm(x,y)
    # gradboost(x,y)
    # labelspreading(x,y,x2,y2)
    x_te = preprocess_1(test)
    y_hat = predict("gb.pkl",x_te)
    print(y_hat)
    y_hat.to_csv("y_hat_gb.csv")