"""
Neural Network + Decision Tree

NeuralNetworkPredictor
  - Predicts ecosystem survival months from 24 parameters
  - Forward prop, backprop, SGD

DecisionTreeAnalyzer
  - Binary classifier: "Survived" (>60 months) vs "Collapsed"
  - Information-gain splitting, feature-importance ranking
  - Extracts human-readable if-then rules
"""

import numpy as np
import random
from simulation import EcosystemParams, run_simulation


# Neural Network Predictor
class NeuralNetworkPredictor:

    def __init__(self, lr: float = 0.001, epochs: int = 80,
                 batch_size: int = 16, callback=None):
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.callback   = callback
        self.trained    = False

        # Architecture: 24 → 64 → 32 → 1
        self._init_weights([24, 64, 32, 1])

        self.loss_history    : list[float] = []
        self.val_loss_history: list[float] = []

        # Normalisation stats
        self._x_mean = None
        self._x_std  = None
        self._y_mean = 0.0
        self._y_std  = 1.0

    # Weight initialisation
    def _init_weights(self, layers):
        self.W, self.b = [], []
        for i in range(len(layers) - 1):
            fan_in, fan_out = layers[i], layers[i + 1]
            std = np.sqrt(2.0 / fan_in)
            self.W.append(np.random.randn(fan_in, fan_out) * std)
            self.b.append(np.zeros((1, fan_out)))

    # Activation functions
    @staticmethod
    def _relu(z):      return np.maximum(0, z)
    @staticmethod
    def _relu_d(z):    return (z > 0).astype(float)

    # Forward pass
    def _forward(self, X):
        self._cache = []
        A = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A @ W + b
            if i < len(self.W) - 1:
                A = self._relu(Z)
            else:
                A = Z           # linear output
            self._cache.append((A, Z))
        return A                # shape (batch, 1)

    # Backward pass
    def _backward(self, X, y_true):
        m   = X.shape[0]
        y_p = self._cache[-1][0]
        dA  = 2 * (y_p - y_true) / m      # MSE gradient

        grads_W, grads_b = [], []
        A_prev = X
        for i in range(len(self.W) - 1, -1, -1):
            Z  = self._cache[i][1]
            if i < len(self.W) - 1:
                dZ = dA * self._relu_d(Z)
            else:
                dZ = dA

            if i > 0:
                A_prev_layer = self._cache[i - 1][0]
            else:
                A_prev_layer = X

            grads_W.insert(0, A_prev_layer.T @ dZ)
            grads_b.insert(0, np.sum(dZ, axis=0, keepdims=True))
            dA = dZ @ self.W[i].T

        for i in range(len(self.W)):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i]  -= self.lr * grads_b[i]

    # Normalise helpers
    def _norm_x(self, X):
        return (X - self._x_mean) / (self._x_std + 1e-8)

    def _norm_y(self, y):
        return (y - self._y_mean) / (self._y_std + 1e-8)

    def _denorm_y(self, y):
        return y * self._y_std + self._y_mean

    # Training
    def train(self, X: np.ndarray, y: np.ndarray,
              val_X: np.ndarray = None, val_y: np.ndarray = None):
        """
        X: parameter vectors
        y: survival months
        """
        self._x_mean = X.mean(axis=0)
        self._x_std  = X.std(axis=0)
        self._y_mean = float(y.mean())
        self._y_std  = float(y.std()) or 1.0

        Xn = self._norm_x(X)
        yn = self._norm_y(y).reshape(-1, 1)
        if val_X is not None:
            vXn = self._norm_x(val_X)
            vyn = self._norm_y(val_y).reshape(-1, 1)

        n = Xn.shape[0]
        for ep in range(self.epochs):
            idx = np.random.permutation(n)
            Xn, yn = Xn[idx], yn[idx]
            for start in range(0, n, self.batch_size):
                Xb = Xn[start: start + self.batch_size]
                yb = yn[start: start + self.batch_size]
                self._forward(Xb)
                self._backward(Xb, yb)

            # Loss
            pred_n  = self._forward(Xn)
            loss    = float(np.mean((pred_n - yn) ** 2))
            self.loss_history.append(loss)

            if val_X is not None:
                vp   = self._forward(vXn)
                vloss= float(np.mean((vp - vyn) ** 2))
                self.val_loss_history.append(vloss)

            if self.callback:
                self.callback(dict(epoch=ep + 1, loss=loss,
                                   val_loss=self.val_loss_history[-1] if self.val_loss_history else None))

        self.trained = True

    # Prediction
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        Xn  = self._norm_x(X)
        out = self._forward(Xn)
        return self._denorm_y(out).flatten()

    def predict_single(self, params: EcosystemParams) -> float:
        vec = params.to_vector().reshape(1, -1)
        return float(np.clip(self.predict(vec)[0], 0, 120))

    # Metrics
    def evaluate(self, X, y):
        preds = self.predict(X)
        mae   = float(np.mean(np.abs(preds - y)))
        ss_res= float(np.sum((y - preds) ** 2))
        ss_tot= float(np.sum((y - y.mean()) ** 2)) or 1.0
        r2    = 1.0 - ss_res / ss_tot
        return dict(mae=mae, r2=r2, predictions=preds)


# Decision Tree Analyser
class _Node:
    __slots__ = ('feature','threshold','left','right','label','samples','impurity')
    def __init__(self):
        self.feature = self.threshold = self.left = self.right = self.label = None
        self.samples  = 0
        self.impurity = 0.0


class DecisionTreeAnalyzer:
    """
    Binary Decision Tree using Information Gain (entropy).
    Predicts: 1 = Survived (>60 months), 0 = Collapsed
    """

    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 8):
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root             = None
        self.feature_names    = EcosystemParams.LABELS
        self.feature_importance: np.ndarray = np.zeros(24)
        self.trained          = False
        self.rules            : list[str] = []
        self.accuracy         = 0.0

    # Entropy
    @staticmethod
    def _entropy(y):
        if len(y) == 0: return 0.0
        p = np.mean(y)
        if p in (0.0, 1.0): return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    # Best split
    def _best_split(self, X, y):
        n_feat  = X.shape[1]
        best_ig = -1.0
        best_f  = best_t = None
        base_e  = self._entropy(y)
        n       = len(y)

        for f in range(n_feat):
            vals = np.unique(X[:, f])
            if len(vals) < 2: continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0

            for t in thresholds:
                left  = y[X[:, f] <= t]
                right = y[X[:, f] >  t]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                ig = base_e - (len(left)/n)*self._entropy(left) \
                            - (len(right)/n)*self._entropy(right)
                if ig > best_ig:
                    best_ig, best_f, best_t = ig, f, t

        return best_f, best_t, best_ig

    # Recursive build
    def _build(self, X, y, depth):
        node           = _Node()
        node.samples   = len(y)
        node.impurity  = self._entropy(y)

        if depth >= self.max_depth or len(y) < self.min_samples_leaf * 2 or node.impurity == 0:
            node.label = int(np.round(np.mean(y))) if len(y) > 0 else 0
            return node

        f, t, ig = self._best_split(X, y)
        if f is None:
            node.label = int(np.round(np.mean(y)))
            return node

        self.feature_importance[f] += ig * len(y)

        mask       = X[:, f] <= t
        node.feature   = f
        node.threshold = t
        node.left  = self._build(X[mask],  y[mask],  depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    # Train
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        X: feature matrix
        y: binary labels (1 = survived, 0 = collapsed)
        """
        self.feature_importance = np.zeros(X.shape[1])
        self.root    = self._build(X, y.astype(int), 0)
        total        = self.feature_importance.sum()
        if total > 0:
            self.feature_importance /= total
        self.trained = True
        self.rules   = self._extract_rules(self.root, [], [])

    # Predict one sample
    def _predict_one(self, x, node):
        if node.label is not None:
            return node.label
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.root) for x in X])

    def evaluate(self, X, y):
        preds = self.predict(X)
        acc   = float(np.mean(preds == y))
        self.accuracy = acc
        return dict(accuracy=acc, predictions=preds)

    # Rule extraction
    def _extract_rules(self, node, conditions, rules):
        if node is None: return rules
        if node.label is not None:
            outcome = "SURVIVED" if node.label == 1 else "COLLAPSED"
            cond    = " AND ".join(conditions) if conditions else "always"
            rules.append(f"IF {cond}  →  {outcome}  ({node.samples} samples)")
            return rules
        fname = self.feature_names[node.feature]
        t     = node.threshold
        self._extract_rules(node.left,
                            conditions + [f"{fname} ≤ {t:.2f}"], rules)
        self._extract_rules(node.right,
                            conditions + [f"{fname} > {t:.2f}"], rules)
        return rules

    def top_features(self, n: int = 8) -> list[tuple]:
        idx = np.argsort(self.feature_importance)[::-1][:n]
        return [(self.feature_names[i], float(self.feature_importance[i]))
                for i in idx]


# Data Collection helper
def collect_training_data(n_samples: int = 200, grid_size: int = 20,
                          hemisphere: str = 'N', max_months: int = 120,
                          callback=None):
    X, y_months = [], []
    for i in range(n_samples):
        params = EcosystemParams.random_params(grid_size)
        months, _, _ = run_simulation(params, grid_size, hemisphere, max_months)
        X.append(params.to_vector())
        y_months.append(float(months))
        if callback:
            callback(i + 1, n_samples)

    X        = np.array(X)
    y_months = np.array(y_months)
    y_binary = (y_months > 60).astype(int)
    return X, y_months, y_binary
