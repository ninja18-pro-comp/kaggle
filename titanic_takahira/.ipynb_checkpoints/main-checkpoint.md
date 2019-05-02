---
jupyter:
  jupytext:
    formats: ipynb,md
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

# Library and etc.

```python
import pandas as pd
import numpy as np
from sklearn import tree

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

def missing_table(df):
    null_val = df.isnull().sum()
    percent = 100 * null_val/len(df)
    missing_data_table = pd.concat([null_val, percent], axis=1)
    missing_data_table_ren_columns = missing_data_table.rename(
            columns = {0: 'missing_data#', 1: '%'})
    return missing_data_table_ren_columns

```

# data set
- trainデータとtestデータについて
  - それぞれ勉強用と実験用的な感じ

## train data

```python
train.head()
```

```python
train.shape
```

```python
train.describe()
```

```python
missing_table(train)
```

## test data

```python
test.head()
```

```python
test.shape
```

```python
test.describe()
```

```python
missing_table(test)
```

# pre-processing data


- In this tutorial, we forcus on 'Age' and 'Embarked'


## train data
- ~~First, we substitute Median of 'Age' for missing data of 'Age'.~~
- First, we substitute Mean of 'Age' for missing data of 'Age'.
- Second, we substitute a value 'S' for missing data of 'Embarked',
  because 'S' is the most popular value in train data.

```python
#train["Age"] = train["Age"].fillna(train["Age"].median())
train["Age"] = train["Age"].fillna(train["Age"].mean())
train["Embarked"] = train["Embarked"].fillna("S")
```

### missing data table(N/A of of 'Age' and 'Embarked' are filled)

```python
missing_table(train)
```

## converting categorical variables of train data from String to int.

```python
train = train.replace("male", 0).replace("female", 1)
train = train.replace("S", 0).replace("C", 1).replace("Q", 2)
train.head(10)
```

### converted data table


### converting test data

```python
test = test["Age"].fillna(test["Age"].mean())
test = test.replace("male", 0).replace("female", 1)
test = test.replace("S", 0).replace("C", 1).replace("Q", 2)
test.Fare[152] = test.Fare.mean()
test.head(10)


```

# processing

```python

target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

my_prediction = my_tree_one.predict(test_features)

print(my_prediction.shape)
print(my_prediction)

PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])

features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

test_features_2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

my_prediction_tree_two = my_tree_two.predict(test_features_2)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution_tree_two = pd.DataFrame(my_prediction_tree_two, PassengerId, columns = ["Survived"])
my_solution_tree_two.to_csv("my_tree_two.csv", index_label = ["PassengerId"])
```

