import unittest
import numpy as np
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from rvfl import RVFLRegressor


def _make_regression(n_samples=200, n_features=10, noise=0.1, random_state=0):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    return train_test_split(X, y, random_state=random_state)


class TestRVFLRegressorInterface(unittest.TestCase):
    """Verify the public API contract."""

    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = _make_regression()
        self.model = RVFLRegressor(n_nodes=100, random_state=0)
        self.model.fit(self.X_train, self.y_train)

    def test_fit_returns_self(self):
        model = RVFLRegressor()
        result = model.fit(self.X_train, self.y_train)
        self.assertIs(result, model)

    def test_predict_shape_1d_target(self):
        preds = self.model.predict(self.X_test)
        self.assertEqual(preds.ndim, 1)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

    def test_predict_shape_2d_target(self):
        y_2d = self.y_train[:, None]
        model = RVFLRegressor(n_nodes=50, random_state=0)
        model.fit(self.X_train, y_2d)
        preds = model.predict(self.X_test)
        # squeezed: single-output should still collapse to 1-D
        self.assertEqual(preds.ndim, 1)

    def test_fitted_attributes_exist(self):
        for attr in ("W_", "b_", "beta_", "scaler_"):
            self.assertTrue(hasattr(self.model, attr), f"Missing attribute: {attr}")

    def test_predict_returns_numpy_array(self):
        preds = self.model.predict(self.X_test)
        self.assertIsInstance(preds, np.ndarray)

    def test_get_set_params(self):
        params = self.model.get_params()
        self.assertIn("n_nodes", params)
        self.assertIn("alpha", params)
        self.model.set_params(n_nodes=50)
        self.assertEqual(self.model.n_nodes, 50)


class TestRVFLRegressorBehaviour(unittest.TestCase):
    """Verify numerical and behavioural correctness."""

    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = _make_regression()

    def test_r2_above_threshold(self):
        model = RVFLRegressor(n_nodes=500, alpha=1e-3, random_state=0)
        model.fit(self.X_train, self.y_train)
        score = r2_score(self.y_test, model.predict(self.X_test))
        self.assertGreater(score, 0.80, f"R² too low: {score:.3f}")

    def test_direct_link_improves_or_matches(self):
        """Direct link should not degrade performance significantly."""
        scores = {}
        for dl in (True, False):
            m = RVFLRegressor(n_nodes=200, direct_link=dl, random_state=0)
            m.fit(self.X_train, self.y_train)
            scores[dl] = r2_score(self.y_test, m.predict(self.X_test))
        # allow a small tolerance; direct_link=True is usually better
        self.assertGreaterEqual(scores[True], scores[False] - 0.05)

    def test_reproducibility(self):
        preds_a = RVFLRegressor(random_state=7).fit(
            self.X_train, self.y_train
        ).predict(self.X_test)
        preds_b = RVFLRegressor(random_state=7).fit(
            self.X_train, self.y_train
        ).predict(self.X_test)
        np.testing.assert_array_equal(preds_a, preds_b)

    def test_different_seeds_differ(self):
        preds_a = RVFLRegressor(random_state=0).fit(
            self.X_train, self.y_train
        ).predict(self.X_test)
        preds_b = RVFLRegressor(random_state=99).fit(
            self.X_train, self.y_train
        ).predict(self.X_test)
        self.assertFalse(np.allclose(preds_a, preds_b))

    def test_higher_alpha_reduces_weight_norm(self):
        def weight_norm(alpha):
            m = RVFLRegressor(n_nodes=100, alpha=alpha, random_state=0)
            m.fit(self.X_train, self.y_train)
            return float(np.linalg.norm(m.beta_))

        self.assertGreater(weight_norm(1e-5), weight_norm(10.0))

    def test_activations(self):
        for act in ("tanh", "relu", "sigmoid"):
            with self.subTest(activation=act):
                m = RVFLRegressor(n_nodes=100, activation=act, random_state=0)
                m.fit(self.X_train, self.y_train)
                preds = m.predict(self.X_test)
                self.assertEqual(preds.shape[0], self.X_test.shape[0])
                self.assertFalse(np.any(np.isnan(preds)))

    def test_no_nan_in_predictions(self):
        model = RVFLRegressor(n_nodes=200, random_state=0)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        self.assertFalse(np.any(np.isnan(preds)))

    def test_multi_output(self):
        X, y_1 = make_regression(n_samples=200, n_features=8, random_state=0)
        y_2 = y_1 * 0.5 + np.random.default_rng(1).normal(size=y_1.shape)
        Y = np.column_stack([y_1, y_2])
        X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, random_state=0)

        model = RVFLRegressor(n_nodes=200, random_state=0)
        model.fit(X_tr, Y_tr)
        preds = model.predict(X_te)
        self.assertEqual(preds.shape, Y_te.shape)


if __name__ == "__main__":
    unittest.main()