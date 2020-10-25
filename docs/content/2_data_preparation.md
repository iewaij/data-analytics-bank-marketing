# Data Preparation


## Import Data

```python
import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option("max_colwidth", None)
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
```

```python
bank_mkt = import_dataset("../data/BankMarketing.csv")
```

## Partition

We need to split the dataset into trainning set and test set, then we train models on the trainning set and only use test set for final validation purposes. However, simply sampling the dataset may lead to unrepresenatative partition given that our dataset is imbalanced and clients have different features. Luckily, `scikit-learn` provides a useful function to select representative data as test data.

```python
from sklearn.model_selection import StratifiedShuffleSplit
```

```python
train_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in train_test_split.split(bank_mkt.drop("y", axis=1), bank_mkt["y"]):
    bank_train_set = bank_mkt.loc[train_index].reset_index(drop=True)
    bank_test_set = bank_mkt.loc[test_index].reset_index(drop=True)
```

## Handling Missing Data



We have several strategies to handle the missing values. For categorical data, we can either treat missing value as a different category or impute them as the most frequent value.

```python
from sklearn.impute import SimpleImputer
```

```python
cat_features = ["job", "marital", "education"]
X = bank_train_set.drop(["duration", "y"], axis=1)
X_cat = X[cat_features]
freq_imp = SimpleImputer(strategy="most_frequent")
freq_imp.fit_transform(X_cat)
```

```python
X_cat = X[cat_features]
fill_imp = SimpleImputer(strategy="constant", fill_value="unknown")
fill_imp.fit_transform(X_cat)
```

Missing values in boolean data is more tricky and requires `pandas` to transform the data first because `SimpleImputer` can not fill nullable boolean data.

```python
bool_features=["default", "housing", "loan"]
X_bool = X[bool_features].astype("category")
freq_imp.fit_transform(X_bool)
```

```python
X_bool = X[bool_features].astype("category")
fill_imp.fit_transform(X_bool)
```

As discussed above, some clients do not have `pdays` but have `poutcome`, which implies that they may have been contacted before but the `pdays` is more than 30 days therefore not inluded. `pdays` can also be cut into different categories which is known as the discretization process.

```python
X_pdays = X["pdays"]
X_pdays[X["pdays"].isna() & X["poutcome"].notna()] = 999
pd.cut(X_pdays, [0, 5, 10, 15, 30, 1000], labels=["pdays<=5", "pdays<=10", "pdays<=15", "pdays<=30", "pdays>30"], include_lowest=True)
```

## Encoding

```python
from sklearn.preprocessing import OneHotEncoder
```

`education`, `month` and `day_of_week` are ordinal data. We can say `basic.6y` is more "advanced" than `basic.4y` for example. Therefore, we should encode `education` into ordinal values or transform them into years of `education`. The same logic also goes for `month` and `day_of_week`. Even though `sklearn` has its own `OrdinalEncoder`, it is using alphabatical order therefore we use pandas instead.

```python
ord_features = ["education", "month", "day_of_week"]
X_ord = X[ord_features]
X_ord.apply(lambda x: x.cat.codes)
```

We will also need `OneHotEncoder` to transform categorical data into multiple binary data.

```python
one_hot_features = ["job", "marital", "default", "housing", "loan"]
one_hot_encoder = OneHotEncoder(drop="first")
X_one_hot = X[one_hot_features].astype("category")
X_one_hot = freq_imp.fit_transform(X_one_hot)
one_hot_encoder.fit_transform(X_one_hot)
one_hot_encoder.get_feature_names(one_hot_features)
```

This can also be done in `pandas`. The advantage of doing one hot encoding in `pandas` is that `pd.get_dummies()` can keep missing values as a row of `0`.


## Transformation Pipeline


We can then wrap all our transformations above into pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
```

```python
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
```

```python
freq_features = ["job", "marital", "education"]

fill_features = ["housing", "loan", "default", "pdays", "poutcome"]

one_hot_features = ["contact"]

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
X_train = preprocessor.fit_transform(bank_train_set.drop(["duration", "y"], axis=1))
y_train = bank_train_set["y"].astype("int").to_numpy()
```

## Baseline Benchmark

```python
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
```

```python
scoring = ["f1", "precision", "recall", "roc_auc"]
# Initialize Model
nb_model = GaussianNB()
logit_model = LogisticRegression(class_weight="balanced")
knn_model = KNeighborsClassifier(n_neighbors=5)
# Train model and get CV results 
nb_cv = cross_validate(nb_model, X_train, y_train, scoring=scoring, cv = 5)
logit_cv = cross_validate(logit_model, X_train, y_train, scoring=scoring, cv = 5)
knn_cv = cross_validate(knn_model, X_train, y_train, scoring=scoring, cv = 5)
# Calculate CV result mean
nb_result = pd.DataFrame(nb_cv).mean().rename("Naive Bayes")
logit_result = pd.DataFrame(logit_cv).mean().rename("Logistic Regression")
knn_result = pd.DataFrame(knn_cv).mean().rename("KNN")
# Store and output result
result = pd.concat([nb_result, logit_result, knn_result], axis=1)
result
```

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
```

```python
X_test = preprocessor.transform(bank_test_set.drop(["duration", "y"], axis=1))
y_test = bank_test_set["y"].astype("int").to_numpy()
# Initialize and fit Model
dummy_model = DummyClassifier(strategy="prior").fit(X_train, y_train)
nb_model = GaussianNB().fit(X_train, y_train)
logit_model = LogisticRegression(class_weight="balanced").fit(X_train, y_train)
knn_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
# Predict and calculate score
dummy_predict = dummy_model.predict(X_test)
dummy_f1 = f1_score(y_test, dummy_predict)
dummy_precision = precision_score(y_test, dummy_predict)
dummy_recall = recall_score(y_test, dummy_predict)
dummy_roc_auc = roc_auc_score(y_test, dummy_predict)
nb_predict = nb_model.predict(X_test)
nb_f1 = f1_score(y_test, nb_predict)
nb_precision = precision_score(y_test, nb_predict)
nb_recall = recall_score(y_test, nb_predict)
nb_roc_auc = roc_auc_score(y_test, nb_predict)
logit_predict = logit_model.predict(X_test)
logit_f1 = f1_score(y_test, logit_predict)
logit_precision = precision_score(y_test, logit_predict)
logit_recall = recall_score(y_test, logit_predict)
logit_roc_auc = roc_auc_score(y_test, logit_predict)
knn_predict = knn_model.predict(X_test)
knn_f1 = f1_score(y_test, knn_predict)
knn_precision = precision_score(y_test, knn_predict)
knn_recall = recall_score(y_test, knn_predict)
knn_roc_auc = roc_auc_score(y_test, knn_predict)
# Store and output result
result = pd.DataFrame(data={"Dummy Classifier": [dummy_f1, dummy_precision, dummy_recall, dummy_roc_auc],
                            "Naive Bayes": [nb_f1, nb_precision, nb_recall, nb_roc_auc],
                            "Logistic Regression": [logit_f1, logit_precision, logit_recall, logit_roc_auc],
                            "KNN": [knn_f1, knn_precision, knn_recall, knn_roc_auc]},
                       index=["F1 Score", "Precision Score", "Recall Score", "ROC AUC Score"])
result
```
