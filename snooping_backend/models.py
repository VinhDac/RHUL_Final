"""models.py - the model zoo (the search space).

A uniform interface over a few sklearn classifiers with different inductive
biases. The from-scratch MLP plugs into the same interface later, as one more
model in the zoo.

sklearn estimators already provide .fit(X, y) and .score(X, y), so the wrapper
stays thin - do not over-build it.

------------------------------------------------------------------------------
Functions to implement (contract - bodies are yours):

  model_zoo() -> dict { name: factory }
      the four families the supervisor named:
        kNN                 -> sklearn.neighbors.KNeighborsClassifier
        decision tree       -> sklearn.tree.DecisionTreeClassifier
        logistic regression -> sklearn.linear_model.LogisticRegression
        SVM                 -> sklearn.svm.SVC

  search_space() -> the configurations to draw from
      a configuration = (model_name, hyperparameters). Start tiny; grow later.

  sample_config(rng) -> one configuration drawn at random from the search space

  build(config) -> an unfitted sklearn estimator for that configuration
------------------------------------------------------------------------------
"""
