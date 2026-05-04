# rvfl

Random Vector Functional Link (RVFL) networks for Python, built on NumPy and
compatible with the scikit-learn API.

RVFL networks are single-hidden-layer feedforward models where the input-to-hidden
weights are drawn randomly and kept fixed. Only the output weights are trained,
via ridge regression in closed form. This makes fitting extremely fast — no
gradient descent, no epochs, no learning rate tuning.

---

## Installation

```bash
pip install rvfl
```

Or in editable mode from source:

```bash
git clone https://github.com/Techtonique/rvfl
cd rvfl
uv pip install -e ".[dev]"
```

**Requirements:** Python >= 3.9, NumPy >= 1.24, scikit-learn >= 1.3.

---

## Quickstart

### Regression

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from rvfl import RVFLRegressor

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = RVFLRegressor(n_nodes=500, alpha=1e-3)
model.fit(X_train, y_train)

print(r2_score(y_test, model.predict(X_test)))
```

### Classification

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rvfl import RVFLClassifier

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = RVFLClassifier(n_nodes=200, activation='relu')
clf.fit(X_train, y_train)

print(accuracy_score(y_test, clf.predict(X_test)))
print(clf.predict_proba(X_test))   # softmax probabilities
```

### Inside a scikit-learn Pipeline

Because both estimators implement `fit` / `predict` and inherit from
`BaseEstimator`, they drop into any sklearn workflow:

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from rvfl import RVFLClassifier

pipe = Pipeline([("clf", RVFLClassifier())])

param_grid = {
    "clf__n_nodes": [100, 500, 1000],
    "clf__alpha":   [1e-4, 1e-2, 1.0],
    "clf__activation": ["tanh", "relu"],
}

gs = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_params_)
```

---

## Parameters

Both `RVFLRegressor` and `RVFLClassifier` share the same constructor:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_nodes` | int | 100 | Number of random hidden neurons |
| `alpha` | float | 1e-3 | Ridge regularisation strength |
| `direct_link` | bool | True | Concatenate raw inputs to hidden features |
| `activation` | str | `'tanh'` | Hidden activation: `'tanh'`, `'relu'`, `'sigmoid'` |
| `scale` | float | 1.0 | Std dev of the random weight distribution |
| `random_state` | int | 42 | Seed for reproducibility |

### Tuning advice

- **`n_nodes`** is the main capacity knob. Start around 200–500; very large
  values rarely hurt thanks to regularisation but do increase memory.
- **`alpha`** behaves like the inverse of `C` in sklearn's `LogisticRegression`.
  If you overfit, raise it; if you underfit, lower it.
- **`direct_link=True`** (the default) almost always helps. It gives the output
  layer access to both the nonlinear hidden features and the original inputs.
- **`activation`** activation function for the hidden layer.

---

## How it works

RVFL replaces gradient-based training with a single linear solve.

1. **Random projection.** Input `X` (standardised) is projected through a
   random matrix `W` and bias `b`, then passed through a nonlinearity to
   produce hidden features `H`.

2. **Optional direct links.** If `direct_link=True`, `X` is appended to `H`,
   giving the output layer both nonlinear and linear access to the inputs.

3. **Ridge regression.** Output weights `beta` are found by solving:

   ```
   min_beta  ||H @ beta - Y||² + alpha * ||beta||²
   ```

   This is solved via an augmented least-squares system rather than the normal
   equations, which is more numerically stable.

4. **Classification.** Targets are one-hot encoded; a softmax is applied to
   the raw outputs at prediction time to produce class probabilities. Training
   still minimises a squared loss on the one-hot matrix — not cross-entropy —
   which preserves the closed-form solve.

---

## Package structure

```
rvfl/
├── rvfl/
│   ├── __init__.py       # public API: RVFLRegressor, RVFLClassifier
│   ├── base.py           # _RVFLBase (shared logic)
│   ├── regressor.py      # RVFLRegressor
│   └── classifier.py     # RVFLClassifier
├── tests/
│   ├── test_regressor.py
│   └── test_classifier.py
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## License

BSD 3-Clause License