import numpy as np
from sklearn.base import RegressorMixin
from .base import _RVFLBase


class RVFLRegressor(_RVFLBase, RegressorMixin):
    """
    Random Vector Functional Link Regressor.

    Fits a single-hidden-layer network with random, fixed weights by solving
    a ridge regression problem in closed form.  Supports single- and
    multi-output regression.

    Inherits all parameters from :class:`_RVFLBase`.

    Examples
    --------
    >>> from rvfl import RVFLRegressor
    >>> model = RVFLRegressor(n_nodes=200, alpha=1e-2)
    >>> model.fit(X_train, y_train)
    RVFLRegressor(alpha=0.01, n_nodes=200)
    >>> y_pred = model.predict(X_test)
    """

    def fit(self, X, Y):
        """
        Fit the RVFL regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs.
        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Training targets.  1-D arrays are treated as single-output.

        Returns
        -------
        self : RVFLRegressor
            Fitted estimator.
        """
        Y = np.array(Y) if np.ndim(Y) == 2 else np.array(Y)[:, None]
        self._fit_hidden(X, Y)
        return self

    def predict(self, X):
        """
        Predict target values for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.  The trailing dimension is squeezed away for
            single-output problems.
        """
        return self._predict_raw(X).squeeze()
