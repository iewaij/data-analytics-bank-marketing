# Introduction

The purpose of Support Vector Machine is to find a hyperplane that distinctly classifies the data points in an N-dimensional space. Hyperplane is the decision boundary that classifies the data points and support vecotrs are the data points that are closer to the hyperplane which influence the position and orientation of the hyperplane.

The worth of the classifier is how well it classifies the unseen data points and therefore our objective is to find a plane with the maximum margin i.e the distance of an observaiton from the hyperplane. The margin brings in extra confidence that the further data points will be classified correctly.

# Theory and Hyperparameter

## Maximum Margin

The understanding of SVM is derived from the loss function of Logistic Regression with l2 regularization:

 $$ 
 J(θ)=\frac 1m \sum_{i=1}^m [ y^{(i)}(−log(p̂^{(i)}))+(1−y^{(i)})(−log(1−p̂^{i}))]+ \frac λ {2m} \sum_{j=1}^nθ_2^{(j)} 
 $$

where
 $$
 p̂^{(i)}=σ(\vecθ^{ T}⋅\vec x^{(i)})=1/(1+e^ {− \vec θ ^T}⋅ \vec x^{(i)})
 $$

In the realm of loss function of Logistic Regression, the individual loss contribution to the overall function is $−log(p̂^{(i)})$ if $y^{(i)}= 1$ and $−log(1−p̂^{(i)})$ if $y^{(i)}= 0$.

By replacing the individual loss contribution to $max(0,1−\vecθ^{ T}⋅\vec x^{(i)})$ and $ max(0,1+\vecθ^{ T}⋅\vec x^{(i)})$ for $y^{(i)}= 1$ and $y^{(i)}= 0$ respectively, SVM penalizes the margin violation more than logistic regression by requiring a prediction bigger than 1 for y =1 and a prediction smaller than -1 if y = 0.


![Screenshot 2020-12-04 at 14.17.12](https://i.imgur.com/4quBUfZ.png)

## Regularizaiton and Trade-off (C parameter)

The regularization term plays the role of widening the distance between the two margins and tells SVM how much we want to avoid the wrong misclassification. A hyperplane with maximal margin might be extremely sensitve to a change in the data points and may lead to overfitting problems.

To achieve the balance of a greater robustness and better classification of the model, we may consider it worthwhile to misclassify a few training data points to do a better job in separating the furture data points.

Hyperparameter C in SVM allows us to dictate the tradeoff between having a wide margin and correctly classiying the training data points. In other words, a large value for C will shrink the margin distance of hyperplane while a small value for C will aim for a larger-margin separator even if it misclassifies some data points.

## Gamma 

Gamma controls how far the influence of a single obeservation  on the decision boundary. The high Gamma indicates only the points closer to the plausible hyperplane are considered and vice versa.

## Kernel

For linearly separable and almost linearly separable data, SVM works well. For data that is not linearly separable, we can project the data to a space where it is linearly separable. What Kernel Trick does is utilizing the existing features and applying some transformations to create new features and calculates the nonlinear decision boundary in higher dimension by using these features.

# GridSearchCV and Results

## SVM in Linear Separable Cases

``` python
linear_svm = LinearSVC(dual=False, class_weight="balanced", random_state=42)

param_distributions = {"loss": ["squared_hinge", "hinge"],
                       "C": loguniform(1e0, 1e3)}

random_search = RandomizedSearchCV(linear_svm,
                                   param_distributions,
                                   scoring="average_precision",
                                   cv=5,
                                   n_jobs=-1,
                                   n_iter=100)

grid_fit = random_search.fit(X_train, y_train)
grid_results = random_search.cv_results_
grid_best_params = random_search.best_params_
grid_best_score = random_search.best_score_

print(f"best parameters found: {grid_best_params}, with mean test score: {grid_best_score}")
```

best parameters found: {'C': 1.1361548657024574, 'loss': 'squared_hinge'}, with mean test score: 0.43354326475418103

```python
param_grid = [
    {"C": [5,2,1]}
    ]
grid_search = GridSearchCV(linear_svm,
                           param_grid,
                           scoring="average_precision",
                           return_train_score=True,
                           cv=5,
                           n_jobs=-1)

grid_fit = grid_search.fit(X_train, y_train)
grid_results = grid_search.cv_results_
grid_best_params = grid_search.best_params_
grid_best_score = grid_search.best_score_

print(f"best parameters found: {grid_best_params}, with mean test score: {grid_best_score}")


linear_svm = LinearSVC(loss="squared_hinge", C=1, dual=False, class_weight="balanced", random_state=42)

```

best parameters found: {'C': 1}, with mean test score: 0.43356240611668306

## Linear SVM Results

![Screenshot 2020-12-04 at 14.45.15](https://i.imgur.com/YGWpNr6.png)

## SVM in Linear Non-separable Cases

We use pipeline to ensure that in the cross validation set, the kernel function is only applied to training fold which is exactly the same fold used for fitting the model. 

We also do a comparison between SGDClassifier and Linear SVC and the latter one gave us slightly better AP rate.

```python
rbf_sgd_clf = Pipeline([
    ("rbf", RBFSampler(random_state=42)),
    ("svm", SGDClassifier(class_weight="balanced"))
])


param_distributions = {
    "rbf__gamma": loguniform(1e-6, 1e-3),
    "svm__alpha": loguniform(1e-10, 1e-6)}

random_search = RandomizedSearchCV(rbf_sgd_clf,
                                   param_distributions,
                                   scoring="average_precision",
                                   cv=5,
                                   n_jobs=-1,
                                   n_iter=10)

grid_fit = random_search.fit(X_train, y_train)
grid_results = random_search.cv_results_
grid_best_params = random_search.best_params_
grid_best_score = random_search.best_score_

print(f"best parameters found: {grid_best_params}, with mean test score: {grid_best_score}")
```

best parameters found: {'rbf__gamma': 0.0007087711398938291, 'svm__alpha': 1.2269339879156183e-07}, with mean test score: 0.4350168498949894

```python

param_grid = {
    "rbf__gamma": [0.0008, 0.0001, 0.001],
    "svm__alpha": [1e-7, 1e-6, 1e-5]}

grid_search = GridSearchCV(rbf_sgd_clf,
                           param_grid,
                           scoring="average_precision",
                           cv=5,
                           n_jobs=-1)

grid_fit = grid_search.fit(X_train, y_train)
grid_results = grid_search.cv_results_
grid_best_params = grid_search.best_params_
grid_best_score = grid_search.best_score_

print(f"best parameters found: {grid_best_params}, with mean test score: {grid_best_score}")
```

best parameters found: {'rbf__gamma': 0.0008, 'svm__alpha': 1e-06}, with mean test score: 0.4403394302575112

```python

rbf_sgd_tuned = rbf_sgd_clf.set_params(rbf__gamma=0.0009, svm__alpha=1e-6)
benchmark(bank_mkt, hot_transformer, rbf_sgd_tuned)
```

![Screenshot 2020-12-04 at 14.47.51](https://i.imgur.com/ARATCWl.png)

```python

rbf_clf = Pipeline([
    ("rbf", RBFSampler(random_state=42)),
    ("svm", LinearSVC(loss="squared_hinge", dual=False, class_weight="balanced", max_iter=1000))
])

param_distributions = {
    "rbf__gamma": loguniform(1e-6, 1e-3),
    "svm__C": loguniform(1e-1, 1e1)}

random_search = RandomizedSearchCV(rbf_clf,
                                   param_distributions,
                                   scoring="average_precision",
                                   cv=5,
                                   n_jobs=-1,
                                   n_iter=10)

grid_fit = random_search.fit(X_train, y_train)
grid_results = random_search.cv_results_
grid_best_params = random_search.best_params_
grid_best_score = random_search.best_score_

print(f"best parameters found: {grid_best_params}, with mean test score: {grid_best_score}")

```

best parameters found: {'rbf__gamma': 0.00026560333125098774, 'svm__C': 6.5900965177317055}, with mean test score: 0.4381080007088255

```python
param_grid = {
    "rbf__gamma": [0.0001, 0.001, 0.01],
    "svm__C": [1, 10, 20]}

grid_search = GridSearchCV(rbf_clf,
                           param_grid,
                           scoring="average_precision",
                           cv=5,
                           n_jobs=-1)

grid_fit = grid_search.fit(X_train, y_train)
grid_results = grid_search.cv_results_
grid_best_params = grid_search.best_params_
grid_best_score = grid_search.best_score_

print(f"best parameters found: {grid_best_params}, with mean test score: {grid_best_score}")
```

best parameters found: {'rbf__gamma': 0.001, 'svm__C': 10}, with mean test score: 0.43986477417883973

```python

rbf_tuned = rbf_clf.set_params(rbf__gamma=0.0009, svm__C=1)

```

![Screenshot 2020-12-04 at 14.48.40](https://i.imgur.com/bjDrpV4.png)

### References

[@hastie_elements_2009]

[@chen_support_2019]

[@patel_chapter_2017]

[@noauthor_machine_nodate]