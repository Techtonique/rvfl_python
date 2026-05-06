import numpy as np
from scipy.special import expit
from sklearn.base import ClassifierMixin
from .base import _RVFLBase


class RVFLClassifier(_RVFLBase, ClassifierMixin):
    """
    Random Vector Functional Link Classifier.

    Solves the regularized normal equation directly on a one-hot encoded
    response matrix of shape ``(n_samples, n_classes)`` — no separate
    regressors, no loops over classes.  All computation is pure matrix
    algebra, identical to what the RVFL regressor does internally except
    that the right-hand side ``Y`` is now a 2-D one-hot matrix instead of
    a 1-D (or single-column) target vector.

    The closed-form solution is:

    .. math::

        \\beta = (H^\\top H + \\alpha I)^{-1} H^\\top Y

    where :math:`H` is the hidden-feature matrix (random projections +
    optional direct link) built by :class:`_RVFLBase`.

    Raw per-class scores ``H @ beta`` are converted to calibrated
    probabilities with the element-wise sigmoid (``expit``) followed by
    row-normalisation, so every row sums to 1.

    Inherits all parameters from :class:`_RVFLBase`.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during ``fit``, in sorted order.
    n_classes_ : int
        Number of unique classes.

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

    def __init__(
        self,
        n_nodes=200,
        activation="relu",
        scale=1.0,
        alpha=1e-4,
        direct_link=True,
        random_state=42,
    ):
        super().__init__(
            n_nodes=n_nodes,
            activation=activation,
            scale=scale,
            alpha=alpha,
            direct_link=direct_link,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _one_hot(y_idx, n_classes):
        """
        Build a one-hot matrix from integer class indices.

        Parameters
        ----------
        y_idx : ndarray of shape (n_samples,)
            Integer class indices in ``{0, ..., n_classes-1}``.
        n_classes : int

        Returns
        -------
        Y : ndarray of shape (n_samples, n_classes), dtype float64
        """
        Y = np.zeros((len(y_idx), n_classes), dtype=float)
        Y[np.arange(len(y_idx)), y_idx] = 1.0
        return Y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the RVFL classifier.

        Builds the hidden-feature matrix ``H`` via :class:`_RVFLBase`,
        one-hot encodes ``y`` into ``Y`` of shape
        ``(n_samples, n_classes)``, then solves the single ridge system

        .. math::

            \\beta = (H^\\top H + \\alpha I)^{-1} H^\\top Y

        to obtain the weight matrix ``beta_`` of shape
        ``(n_hidden_features, n_classes)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs.
        y : array-like of shape (n_samples,)
            Class labels (integers, strings, or any type sortable by
            ``numpy.unique``).

        Returns
        -------
        self : RVFLClassifier
            Fitted estimator.
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Map arbitrary labels → contiguous integers 0 … n_classes-1
        label_to_idx = {lbl: i for i, lbl in enumerate(self.classes_)}
        y_idx = np.array([label_to_idx[lbl] for lbl in y])

        # One-hot response matrix  (n_samples × n_classes)
        Y = self._one_hot(y_idx, self.n_classes_)

        # Delegate to _RVFLBase: builds H and solves the normal equation
        # for a multi-column right-hand side — identical path to regression.
        self._fit_hidden(X, Y)
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for ``X``.

        Raw scores ``H @ beta_`` (shape ``(n_samples, n_classes)``) are
        passed through the element-wise sigmoid and then row-normalised:

        .. math::

            p_{ij} = \\frac{\\sigma(s_{ij})}{\\sum_k \\sigma(s_{ik})}

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Estimated class probabilities.  Rows sum to 1.
        """
        scores = self._predict_raw(X)  # (n_samples, n_classes)
        proba = expit(scores)  # element-wise sigmoid
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-10, 1e-10, row_sums)
        return proba / row_sums

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
            Predicted class labels in the same encoding seen during ``fit``.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def decision_function(self, X):
        """
        Raw per-class scores before sigmoid.

        For binary problems returns a 1-D array (score for the positive
        class); for multiclass returns a 2-D array of shape
        ``(n_samples, n_classes)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
        """
        scores = self._predict_raw(X)
        if self.n_classes_ == 2:
            return scores[:, 1]
        return scores
