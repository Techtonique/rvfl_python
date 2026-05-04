import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from .base import _RVFLBase


class RVFLClassifier(_RVFLBase, ClassifierMixin):
    """
    Random Vector Functional Link Classifier.

    Targets are one-hot encoded via :class:`sklearn.preprocessing.LabelBinarizer`
    before solving the ridge system.  Predictions are obtained by applying a
    softmax to the raw outputs and returning the argmax class.

    Supports binary and multiclass problems.  The class order follows
    ``LabelBinarizer``, so it is consistent with the order seen during
    ``fit``.

    Inherits all parameters from :class:`_RVFLBase`.

    Attributes
    ----------
    lb_ : LabelBinarizer
        Fitted binarizer used to encode targets and decode predictions.

    Examples
    --------
    >>> from rvfl import RVFLClassifier
    >>> clf = RVFLClassifier(n_nodes=200, activation='relu')
    >>> clf.fit(X_train, y_train)
    RVFLClassifier(activation='relu', n_nodes=200)
    >>> clf.predict(X_test)
    array([...])
    >>> clf.predict_proba(X_test)
    array([[...]])
    """
    def __init__(self, n_nodes=200, 
                 activation="relu", scale=1.0, 
                 alpha=1e-4, direct_link=True,
                 random_state=42):
        super().__init__(n_nodes=n_nodes,        
        activation=activation,
        scale=scale,
        alpha=alpha,
        direct_link=direct_link,
        random_state=random_state)
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the RVFL classifier.

        Internally one-hot encodes ``y``, expands binary targets to two
        columns so that softmax is well-defined, then delegates to
        :meth:`_fit_hidden`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs.
        y : array-like of shape (n_samples,)
            Class labels.  Can be integers, strings, or any type accepted
            by :class:`sklearn.preprocessing.LabelBinarizer`.

        Returns
        -------
        self : RVFLClassifier
            Fitted estimator.
        """
        self.lb_ = LabelBinarizer()
        Y = self.lb_.fit_transform(y).astype(float)
        self.classes_ = self.lb_.classes_
        if Y.shape[1] == 1:  # binary: expand to two columns
            Y = np.hstack([1 - Y, Y])
        self._fit_hidden(X, Y)
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities using a softmax over the raw outputs.

        The softmax is computed with the standard max-subtraction trick for
        numerical stability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Estimated class probabilities.  Rows sum to 1.
        """
        logits = self._predict_raw(X)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        """
        Predict class labels for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class labels, in the same dtype / encoding as the
            labels seen during ``fit``.
        """
        return self.lb_.classes_[self.predict_proba(X).argmax(axis=1)]
