import unittest
import numpy as np
from sklearn.datasets import make_classification, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from rvfl import RVFLClassifier


def _make_binary(n_samples=300, n_features=10, random_state=0):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=2,
        random_state=random_state,
    )
    return train_test_split(X, y, random_state=random_state)


def _make_multiclass(random_state=0):
    X, y = load_wine(return_X_y=True)
    return train_test_split(X, y, random_state=random_state)


class TestRVFLClassifierInterface(unittest.TestCase):
    """Verify the public API contract."""

    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = _make_binary()
        self.clf = RVFLClassifier(n_nodes=100, random_state=0)
        self.clf.fit(self.X_train, self.y_train)

    def test_fit_returns_self(self):
        clf = RVFLClassifier()
        result = clf.fit(self.X_train, self.y_train)
        self.assertIs(result, clf)

    def test_predict_shape(self):
        preds = self.clf.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])

    def test_predict_proba_shape(self):
        proba = self.clf.predict_proba(self.X_test)
        self.assertEqual(proba.shape, (self.X_test.shape[0], 2))

    def test_predict_proba_sums_to_one(self):
        proba = self.clf.predict_proba(self.X_test)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(self.X_test)), atol=1e-6)

    def test_predict_proba_in_range(self):
        proba = self.clf.predict_proba(self.X_test)
        self.assertTrue(np.all(proba >= 0))
        self.assertTrue(np.all(proba <= 1))

    def test_predict_consistent_with_proba(self):
        proba = self.clf.predict_proba(self.X_test)
        expected = self.clf.classes_[proba.argmax(axis=1)]  # noqa: F841
        # predict() should agree with argmax of predict_proba()
        np.testing.assert_array_equal(
            self.clf.predict(self.X_test),
            self.clf.classes_[proba.argmax(axis=1)],
        )

    def test_fitted_attributes_exist(self):
        for attr in ("W_", "b_", "beta_", "scaler_"):
            self.assertTrue(hasattr(self.clf, attr), f"Missing attribute: {attr}")

    def test_predict_returns_numpy_array(self):
        preds = self.clf.predict(self.X_test)
        self.assertIsInstance(preds, np.ndarray)

    def test_get_set_params(self):
        params = self.clf.get_params()
        self.assertIn("n_nodes", params)
        self.clf.set_params(alpha=0.1)
        self.assertEqual(self.clf.alpha, 0.1)


class TestRVFLClassifierBehaviour(unittest.TestCase):
    """Verify numerical and behavioural correctness."""

    def test_binary_accuracy_above_threshold(self):
        X_train, X_test, y_train, y_test = _make_binary()
        clf = RVFLClassifier(n_nodes=300, alpha=1e-3, random_state=0)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        self.assertGreater(acc, 0.80, f"Binary accuracy too low: {acc:.3f}")

    def test_multiclass_accuracy_above_threshold(self):
        X_train, X_test, y_train, y_test = _make_multiclass()
        clf = RVFLClassifier(n_nodes=300, alpha=1e-3, random_state=0)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        self.assertGreater(acc, 0.90, f"Multiclass accuracy too low: {acc:.3f}")

    def test_multiclass_proba_shape(self):
        X_train, X_test, y_train, y_test = _make_multiclass()
        clf = RVFLClassifier(n_nodes=100, random_state=0)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        n_classes = len(np.unique(y_train))
        self.assertEqual(proba.shape, (X_test.shape[0], n_classes))

    def test_multiclass_proba_sums_to_one(self):
        X_train, X_test, y_train, _ = _make_multiclass()
        clf = RVFLClassifier(n_nodes=100, random_state=0)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_test)), atol=1e-6)

    def test_reproducibility(self):
        X_train, X_test, y_train, _ = _make_binary()
        preds_a = RVFLClassifier(random_state=7).fit(X_train, y_train).predict(X_test)
        preds_b = RVFLClassifier(random_state=7).fit(X_train, y_train).predict(X_test)
        np.testing.assert_array_equal(preds_a, preds_b)

    def test_different_seeds_differ(self):
        X_train, X_test, y_train, _ = _make_binary()
        preds_a = RVFLClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
        preds_b = RVFLClassifier(random_state=99).fit(X_train, y_train).predict(X_test)
        self.assertFalse(np.array_equal(preds_a, preds_b))

    def test_string_labels(self):
        X_train, X_test, y_train, _ = _make_binary()
        y_str = np.where(y_train == 0, "cat", "dog")
        clf = RVFLClassifier(n_nodes=100, random_state=0)
        clf.fit(X_train, y_str)
        preds = clf.predict(X_test)
        self.assertTrue(set(preds).issubset({"cat", "dog"}))

    def test_activations(self):
        X_train, X_test, y_train, y_test = _make_binary()
        for act in ("tanh", "relu", "sigmoid"):
            with self.subTest(activation=act):
                clf = RVFLClassifier(n_nodes=100, activation=act, random_state=0)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                self.assertEqual(preds.shape[0], X_test.shape[0])
                proba = clf.predict_proba(X_test)
                self.assertFalse(np.any(np.isnan(proba)))

    def test_no_nan_in_proba(self):
        X_train, X_test, y_train, _ = _make_binary()
        clf = RVFLClassifier(n_nodes=200, random_state=0)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        self.assertFalse(np.any(np.isnan(proba)))

    def test_direct_link_false(self):
        X_train, X_test, y_train, y_test = _make_binary()
        clf = RVFLClassifier(n_nodes=200, direct_link=False, random_state=0)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        self.assertGreater(acc, 0.70)


if __name__ == "__main__":
    unittest.main()