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

```python
import pandas as pd
import numpy as np

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

def missing_table(df):
    null_val = df.isnull().sum()
    percent = 100 * null_val/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
            columns = {0: '欠損数', 1: '%'})
    return kesson_table_ren_columns

```

# train data
## list

```python
train.head()
```

## shape

```python
train.shape
```

## describe

```python
train.describe()
```

## missing data table

```python
missing_table(train)
```

# test data


## list

```python
test.head()
```

## shape

```python
test.shape
```

## describe

```python
test.describe()
```

## missing data table

```python
missing_table(test)
```

# pre-processing data


- In this tutorial, we forcus on 'Age' and 'Embarked'


## train data
- First, we substitute Median of 'Age' for missing data of 'Age'.
- Second, we substitute a value 'S' for missing data of 'Embarked',
  because 'S' is the most popular value in train data.

```python
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")
```

### missing data table(N/A of of 'Age' and 'Embarked' are filled)

```python
missing_table(train)
```

## converting categorical variables from String to int.

```python
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
```

### converted data table

```python
train.head(10)
```

### converting test data

```python
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.median()
print(train.head(10))

from sklearn import tree
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

# test jupytext

```python
train.head(10)
```
