import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator


class _RVFLBase(BaseEstimator):
    """
    Base class for Random Vector Functional Link (RVFL) networks.

    RVFL networks are single-hidden-layer feedforward networks where the
    hidden-layer weights are drawn randomly and kept fixed. Only the output
    weights are trained, via ridge regression in closed form. This makes
    training extremely fast compared to gradient-based alternatives.

    The augmented-system formulation used here is numerically equivalent to
    the normal equations with L2 regularisation but avoids forming ``H.T @ H``
    explicitly, which improves numerical stability for wide feature matrices.

    Parameters
    ----------
    n_nodes : int, default=100
        Number of randomly initialised hidden neurons.
    alpha : float, default=1e-3
        Ridge (L2) regularisation strength. Larger values shrink the output
        weights more aggressively and reduce variance at the cost of bias.
    direct_link : bool, default=True
        If ``True``, the original (scaled) inputs are concatenated to the
        hidden-layer activations before solving for the output weights.
        Direct links often improve accuracy at negligible cost.
    activation : {'tanh', 'relu', 'sigmoid'}, default='tanh'
        Element-wise activation function applied to the hidden layer.
    scale : float, default=1.0
        Standard deviation of the zero-mean Gaussian used to sample the
        random hidden weights ``W_`` and biases ``b_``.
    random_state : int, default=42
        Seed passed to ``numpy.random.RandomState`` for reproducibility.

    Attributes
    ----------
    W_ : ndarray of shape (n_features, n_nodes)
        Random input-to-hidden weight matrix, fixed after ``fit``.
    b_ : ndarray of shape (n_nodes,)
        Random hidden biases, fixed after ``fit``.
    beta_ : ndarray of shape (n_nodes [+ n_features], n_outputs)
        Learned output weights.  The leading dimension is
        ``n_nodes + n_features`` when ``direct_link=True``.
    scaler_ : StandardScaler
        Fitted scaler used to standardise inputs.
    """

    def __init__(
        self,
        n_nodes=100,
        alpha=1e-3,
        direct_link=True,
        activation="tanh",
        scale=1.0,
        random_state=42,
    ):
        self.n_nodes = n_nodes
        self.alpha = alpha
        self.direct_link = direct_link
        self.activation = activation
        self.scale = scale
        self.random_state = random_state

    def _activate(self, Z):
        """
        Apply the chosen element-wise activation function.

        Parameters
        ----------
        Z : ndarray of shape (n_samples, n_nodes)
            Pre-activation matrix.

        Returns
        -------
        ndarray of shape (n_samples, n_nodes)
            Post-activation matrix.
        """
        if self.activation == "relu":
            return np.maximum(0, Z)
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        return np.tanh(Z)

    def _hidden(self, X):
        """
        Compute the (extended) hidden representation of ``X``.

        Applies the random projection, adds the bias, activates, and
        optionally appends the raw inputs as direct links.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Standardised input matrix.

        Returns
        -------
        H : ndarray of shape (n_samples, n_nodes [+ n_features])
            Hidden (+ direct-link) feature matrix.
        """
        H = self._activate(X @ self.W_ + self.b_)
        return np.hstack([H, X]) if self.direct_link else H

    def _init_random(self, n_in):
        """
        Draw and store the random hidden weights and biases.

        Parameters
        ----------
        n_in : int
            Number of input features (i.e. columns of ``W_``).
        """
        rng = np.random.RandomState(self.random_state)
        self.W_ = rng.normal(0, self.scale, (n_in, self.n_nodes))
        self.b_ = rng.normal(0, self.scale, (self.n_nodes,))

    def _fit_hidden(self, X, Y):
        """
        Standardise inputs, build the hidden representation, and solve
        for the output weights via ridge regression.

        The ridge problem is cast as an ordinary least-squares problem on an
        augmented system::

            [H          ] beta = [Y         ]
            [sqrt(alpha)*I]       [0         ]

        so that ``numpy.linalg.lstsq`` can be used directly without forming
        the normal equations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw training inputs.
        Y : ndarray of shape (n_samples, n_outputs)
            Target matrix (already one-hot encoded for classifiers).
        """
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X)

        self._init_random(X.shape[1])
        H = self._hidden(X)

        p = H.shape[1]
        H_aug = np.vstack([H, np.sqrt(self.alpha) * np.eye(p)])
        Y_aug = np.vstack([Y, np.zeros((p, Y.shape[1]))])

        self.beta_ = np.linalg.lstsq(H_aug, Y_aug, rcond=None)[0]

    def _predict_raw(self, X):
        """
        Compute raw (pre-activation) model outputs for new inputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw input matrix.

        Returns
        -------
        ndarray of shape (n_samples, n_outputs)
            Linear combination of the hidden (+ direct-link) features and
            the learned output weights.
        """
        return self._hidden(self.scaler_.transform(X)) @ self.beta_
