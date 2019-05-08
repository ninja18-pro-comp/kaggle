---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 1.0.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

refer to https://lp-tech.net/articles/0QUUd

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.head(10) # raw train data
```

- total 891
- missing data
  - Age (714/891)
  - Cabin (204/891)
  - Embarked (889/891)

```python
train.info()
```

# replace strings with numbers

```python
train = train.replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
test = test.replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
```

# fill N/A data by mean

```python
train["Age"].fillna(train.Age.mean(), inplace=True)
train["Embarked"].fillna(train.Embarked.mean(), inplace=True) #平均で埋めていいの？
train.head()
```

# Classify Name

```python
#pd.set_option("display.max_rows", 1000)
combine1 = [train]

for train in combine1:
    train['Salutation'] = train['Name'].str.extract(' ([A-Za-z]+).', expand=False)
#print(train['Salutation'])
counts = train['Salutation'].value_counts()
print(counts)
```

```python
train['Salutation'].replace('Mlle')
```

```python
combine1 = [train]

for train in combine1:
    train['Salutation'] = train.Name.str.extract(' ([A-Za-z]+).', expand=False)
for train in combine1:
    train['Salutation'] = train['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    train['Salutation'] = train['Salutation'].replace('Mlle', 'Miss')
    train['Salutation'] = train['Salutation'].replace('Ms', 'Miss')
    train['Salutation'] = train['Salutation'].replace('Mme', 'Mrs')
    del train['Name']
Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 
for train in combine1: 
    train['Salutation'] = train['Salutation'].map(Salutation_mapping) 
    train['Salutation'] = train['Salutation'].fillna(0)

for train in combine1: 
    train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])
    train['Ticket_Lett'] = train['Ticket_Lett'].apply(lambda x: str(x)) 
    train['Ticket_Lett'] = np.where((train['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), train['Ticket_Lett'], np.where((train['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0')) 
    train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x)) 
    del train['Ticket'] 
train['Ticket_Lett']=train['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3)

for train in combine1: 
    train['Cabin_Lett'] = train['Cabin'].apply(lambda x: str(x)[0]) 
    train['Cabin_Lett'] = train['Cabin_Lett'].apply(lambda x: str(x)) 
    train['Cabin_Lett'] = np.where((train['Cabin_Lett']).isin([ 'F', 'E', 'D', 'C', 'B', 'A']),train['Cabin_Lett'], np.where((train['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0'))
    del train['Cabin'] 
train['Cabin_Lett']=train['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1)

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
for train in combine1:
    train['IsAlone'] = 0
    train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
train.head(10)
```

```python
train_data = train.values
xs = train_data[:, 2:] # Pclass以降の変数
y  = train_data[:, 1]  # 正解データ
```

# testデータ加工

```python
test.info()
```

```python
test["Age"].fillna(train.Age.mean(), inplace=True)
test["Fare"].fillna(train.Fare.mean(), inplace=True)

combine = [test]
for test in combine:
    test['Salutation'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for test in combine:
    test['Salutation'] = test['Salutation'].replace(['Lady', 'Countess','Capt', 'Col',\
         'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    test['Salutation'] = test['Salutation'].replace('Mlle', 'Miss')
    test['Salutation'] = test['Salutation'].replace('Ms', 'Miss')
    test['Salutation'] = test['Salutation'].replace('Mme', 'Mrs')
    del test['Name']
Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for test in combine:
    test['Salutation'] = test['Salutation'].map(Salutation_mapping)
    test['Salutation'] = test['Salutation'].fillna(0)

for test in combine:
        test['Ticket_Lett'] = test['Ticket'].apply(lambda x: str(x)[0])
        test['Ticket_Lett'] = test['Ticket_Lett'].apply(lambda x: str(x))
        test['Ticket_Lett'] = np.where((test['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), test['Ticket_Lett'],
                                   np.where((test['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            '0', '0'))
        test['Ticket_Len'] = test['Ticket'].apply(lambda x: len(x))
        del test['Ticket']
test['Ticket_Lett']=test['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3) 

for test in combine:
        test['Cabin_Lett'] = test['Cabin'].apply(lambda x: str(x)[0])
        test['Cabin_Lett'] = test['Cabin_Lett'].apply(lambda x: str(x))
        test['Cabin_Lett'] = np.where((test['Cabin_Lett']).isin(['T', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']),test['Cabin_Lett'],
                                   np.where((test['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            '0','0'))        
        del test['Cabin']
test['Cabin_Lett']=test['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1).replace("G",1) 

test["FamilySize"] = train["SibSp"] + train["Parch"] + 1

for test in combine:
    test['IsAlone'] = 0
    test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1
    
test_data = test.values
xs_test = test_data[:, 1:]

test.head()
```

# 機械学習
## ランダムフォレストを利用

```python
from sklearn.ensemble import RandomForestClassifier

random_forest=RandomForestClassifier()
random_forest.fit(xs, y) # xs : Pclass以降の変数、 y: Survived
Y_pred = random_forest.predict(xs_test)

import csv
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), Y_pred.astype(int)):
        writer.writerow([pid, survived])
```

## ランダムフォレストのパラメータをチューニング

```python
random_forest=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=25, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=15,
            min_weight_fraction_leaf=0.0, n_estimators=51, n_jobs=4,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
random_forest.fit(xs, y) # xs : Pclass以降の変数、 y: Survived
Y_pred = random_forest.predict(xs_test)

import csv
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), Y_pred.astype(int)):
        writer.writerow([pid, survived])
```

# データ分析
性別によってどれほど生存に違いが出たか?
0が男性、1が女性
女性の生存率が高い。

```python
import matplotlib.pyplot as plt
import seaborn as sns

g = sns.catplot(x="Sex", y="Survived",  data=train,
                   height=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
```

そして男性の人数がかなり多い

乗組員に男性が多かったから? <= 全てにPclassあるので乗組員は含まれていない

```python
g = sns.countplot(x='Sex', data=train)
```

乗っていた等級による生存率

等級がいい順に生存率が高い


```python
g = sns.catplot(x="Pclass",y="Survived",data=train,kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
```

等級と性別を複合した時の生存率

1等と2等の女性の生存率が高く、2等と3等の男性の生存率が低いです。

```python
g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   height=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
```

敬称による生存率

"others" : 0, "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5

1のMrは大人の男性　-> 生存率が少ない

2と3はMissとMrsなので女性 -> 生存率が高い

4はMaster、青年や若い男性

```python
g = sns.catplot(x="Salutation", y="Survived",  data=train,
                   height=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
```

データの生存率の相関

- 性別(Sex,Salutation)による相関、属している社会階級つまりお金をどれだけ持っているか(Pclass,Fare)に対する相関が高い
- Cabin_LettとTicket_Lettの相関も高い <= 社会的地位？
- 他にはIsAloneも相関が高い

```python
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
del train['PassengerId']
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
```

一緒に乗船した人数による生存率

一人で乗っていた人が多い

1人か5人以上で乗っていると生存率が悪い　<= 大家族であると当然お金もたくさんかかる.3等に乗った人が多い.一人の方々も3等が多い。3等なので救出の優先度も低い

```python
sns.countplot(x='FamilySize', data = train, hue = 'Survived')
```

```python
sns.countplot(x='FamilySize', data = train,hue = 'Pclass')
```

乗船場所による生存率の違い

タイタニックの航路はイギリスのサウサンプトン→フランスのシェルブール→アイルランドのクイーンズタウンの順番

シェルブールから乗った人の生存率が高い　＜＝1等に乗った人の割合が高かった

クイーンズタウンから乗った人は3等ばかりなのに生存率が少し高い　＜＝　男女比？

```python
t=pd.read_csv("train.csv").replace("S",0).replace("C",1).replace("Q",2)
train['Embarked']= t['Embarked']
g = sns.catplot(x="Embarked", y="Survived",  data=train,
                   height=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
```

```python
g = sns.countplot(x='Embarked', data = train,hue = 'Pclass')
```

```python
sns.countplot(x='Embarked', data = train,hue = 'Sex')
```

年齢による違い

10代後半から30代ほどまでは死亡率が高い

子供の死亡率は低い

15歳より上だとほとんど成人とみなされていた

老人の死亡率も高い

```python
plt.figure()
sns.FacetGrid(data=t, hue="Survived", aspect=4).map(sns.kdeplot, "Age", shade=True)
plt.ylabel('Passenger Density')
plt.title('KDE of Age against Survival')
plt.legend()
```

```python
for t in combine1: 
    t.loc[ t['Age'] <= 15, 'Age'] = 0
    t.loc[(t['Age'] > 15) & (t['Age'] <= 25), 'Age'] = 1
    t.loc[(t['Age'] > 25) & (t['Age'] <= 48), 'Age'] = 2
    t.loc[(t['Age'] > 48) & (t['Age'] <= 64), 'Age'] = 3
    t.loc[ t['Age'] > 64, 'Age'] =4
g = sns.catplot(x="Age", y="Survived",  data=t,
                   height=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
```

男女比と生存の数がほぼ一緒

```python
g = sns.countplot(x='Age', data = t,hue = 'Sex')
```

```python
sns.countplot(x='Age', data = t,hue = 'Survived')
```
