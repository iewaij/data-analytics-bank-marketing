# Neural Network


## Data Preparation

```python
import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option("max_colwidth", None)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
```

```python
# Features where the missing values will be imputed as the most frequent value
freq_features = ["job", "marital", "education"]

# Features where the missing values will be filled as a distinct value
fill_features = ["housing", "loan", "default", "pdays", "poutcome"]

# Features that are not in freq_features or fill_features but need to be one hot encoded
one_hot_features = ["contact"]
```

```python
def import_dataset(filename):
    bank_mkt = pd.read_csv(filename,
                           na_values=["unknown", "nonexistent"],
                           true_values=["yes", "success"],
                           false_values=["no", "failure"])
    # Treat pdays = 999 as missing values
    bank_mkt["pdays"] = bank_mkt["pdays"].replace(999, pd.NA)
    # Convert types, "Int64" is nullable integer data type in pandas
    bank_mkt = bank_mkt.astype(dtype={"age": "Int64",
                                      "job": "category",
                                      "marital": "category",
                                      "education": "category",
                                      "default": "boolean",
                                      "housing": "boolean",
                                      "loan": "boolean",
                                      "contact": "category",
                                      "month": "category",
                                      "day_of_week": "category",
                                      "duration": "Int64",
                                      "campaign": "Int64",
                                      "pdays": "Int64",
                                      "previous": "Int64",
                                      "poutcome": "boolean",
                                      "y": "boolean"})
    # reorder categorical data
    bank_mkt["education"] = bank_mkt["education"].cat.reorder_categories(["illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school", "professional.course", "university.degree"], ordered=True)
    bank_mkt["month"] = bank_mkt["month"].cat.reorder_categories(["mar", "apr", "jun", "jul", "may", "aug", "sep", "oct", "nov", "dec"], ordered=True)
    bank_mkt["day_of_week"] = bank_mkt["day_of_week"].cat.reorder_categories(["mon", "tue", "wed", "thu", "fri"], ordered=True)
    return bank_mkt

def pdays_transformation(X):
    """Feature Engineering `pdays`."""
    X = X.copy()
    X.loc[X["pdays"].isna() & X["poutcome"].notna(), "pdays"] = 999
    X["pdays"] = pd.cut(X["pdays"], [0, 5, 10, 15, 30, 1000], labels=["<=5", "<=10", "<=15", "<=30", ">30"], include_lowest=True)
    return X

def ordinal_transformation(X, education=None):
    """Encode ordinal labels.

    education: if education is "year", education column will be encoded into years of eductaion.
    """
    X = X.copy()
    ordinal_features = ["education", "month", "day_of_week"]
    X[ordinal_features] = X[ordinal_features].apply(lambda x: x.cat.codes)
    if education=="year":
        education_map = { 0: 0, # illiterate
                          1: 4, # basic.4y
                          2: 6, # basic.6y
                          3: 9, # basic.9y
                          4: 12, # high.school
                          5: 15, # professional course
                          6: 16} # university
        X["education"] = X["education"].replace(education_map)
    return X

def bool_transformation(X):
    """Transform boolean data into categorical data."""
    X = X.copy()
    bool_features = ["default", "housing", "loan", "poutcome"]
    X[bool_features] = X[bool_features].astype("category")
    X[bool_features] = X[bool_features].replace({True: "true", False: "false"})
    return X

cut_transformer = FunctionTransformer(pdays_transformation)

ordinal_transformer = FunctionTransformer(ordinal_transformation)

bool_transformer = FunctionTransformer(bool_transformation)

freq_transformer = Pipeline([
    ("freq_imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encoder", OneHotEncoder(drop="first", handle_unknown="error"))
])

fill_transformer = Pipeline([
    ("freq_imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("one_hot_encoder", OneHotEncoder(drop="first", handle_unknown="error"))
])

cat_transformer = ColumnTransformer([
    ("freq_imputer", freq_transformer, freq_features),
    ("fill_imputer", fill_transformer, fill_features),
    ("one_hot_encoder", OneHotEncoder(drop="first", handle_unknown="error"), one_hot_features)
], remainder="passthrough")

preprocessor = Pipeline([
    ("bool_transformer", bool_transformer),
    ("cut_transformer", cut_transformer),
    ("ordinal_transformer", ordinal_transformer),
    ("cat_transformer", cat_transformer),
    ("scaler", StandardScaler())
])
```

```python
bank_mkt = import_dataset("../data/BankMarketing.csv")

train_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in train_test_split.split(bank_mkt.drop("y", axis=1), bank_mkt["y"]):
    bank_train_set = bank_mkt.loc[train_index].reset_index(drop=True)
    bank_test_set = bank_mkt.loc[test_index].reset_index(drop=True)

X_train = preprocessor.fit_transform(bank_train_set.drop(["duration", "y"], axis=1))
y_train = bank_train_set["y"].astype("int").to_numpy()
```

## Methods

```python

```
