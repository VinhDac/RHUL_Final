"""models.py - the sklearn garden (fixed-setting classifiers).

These are NOT the searched instrument (that is the PyTorch MLP in mlp.py). They are
four classifiers with different inductive biases, run at FIXED settings, used only for
two side demonstrations:
  * Case 4 (XOR), Core.md section 5: the LINEAR models (logreg, linear SVM) sit at
    chance while the MLP learns the nonlinear rule - so the deep model earns its place.
  * the isometry control (appendix): rotating the data leaves every coordinate-free
    method (kNN, logreg, linear SVM) untouched and only the axis-aligned tree drops.

sklearn estimators already give .fit(X, y) / .score(X, y); the wrapper stays thin.
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def model_zoo():
    """The four families the supervisor named, at fixed settings -> {name: fresh estimator}.

    A fresh (unfitted) estimator each call. kNN and the (unpruned) tree can fit a
    nonlinear rule; logistic regression and the LINEAR SVM cannot - that linear /
    nonlinear split is exactly the point of the Case-4 demo.
    """
    return {
        "kNN":    KNeighborsClassifier(n_neighbors=5),   # distance-based (coordinate-free)
        "tree":   DecisionTreeClassifier(random_state=0),  # axis-aligned splits
        "logreg": LogisticRegression(max_iter=1000),     # linear boundary
        "SVM":    SVC(kernel="linear"),                  # linear boundary
    }


LINEAR = ("logreg", "SVM")   # the linear baselines compared against the MLP in Case 4
