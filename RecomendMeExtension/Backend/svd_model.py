"""
Pure NumPy/scipy SVD collaborative filtering — no scikit-surprise dependency.
Drop-in replacement for surprise.SVD with the same .predict(uid, iid).est interface.
"""
import numpy as np


class _Prediction:
    __slots__ = ("est",)

    def __init__(self, est: float):
        self.est = est


class SVDModel:
    """
    Stochastic Gradient Descent matrix factorization (same algorithm as surprise.SVD).
    r̂_ui = μ + b_u + b_i + q_i · p_u
    """

    def __init__(self, n_factors=40, n_epochs=100, lr_all=0.005, reg_all=0.15, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state

    def fit(self, ratings):
        """
        ratings: list/sequence of (uid, iid, rating) tuples — strings for uid/iid, float for rating.
        """
        rng = np.random.RandomState(self.random_state)
        ratings = list(ratings)

        users = sorted({r[0] for r in ratings})
        items = sorted({r[1] for r in ratings})

        self._user_idx = {u: i for i, u in enumerate(users)}
        self._item_idx = {it: i for i, it in enumerate(items)}

        n_u = len(users)
        n_i = len(items)
        k = self.n_factors
        lr = self.lr_all
        reg = self.reg_all

        self.global_mean_ = float(np.mean([r[2] for r in ratings]))

        bu = np.zeros(n_u)
        bi = np.zeros(n_i)
        pu = rng.normal(0, 0.1, (n_u, k))
        qi = rng.normal(0, 0.1, (n_i, k))

        for _ in range(self.n_epochs):
            rng.shuffle(ratings)
            for uid, iid, r in ratings:
                u = self._user_idx[uid]
                i = self._item_idx[iid]
                err = r - (self.global_mean_ + bu[u] + bi[i] + pu[u] @ qi[i])
                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])
                pu_u = pu[u].copy()
                pu[u] += lr * (err * qi[i] - reg * pu[u])
                qi[i] += lr * (err * pu_u - reg * qi[i])

        self._bu = bu
        self._bi = bi
        self._pu = pu
        self._qi = qi
        return self

    def predict(self, uid, iid):
        """Return a _Prediction with .est clamped to [1, 5]."""
        gm = self.global_mean_
        u_idx = self._user_idx.get(uid)
        i_idx = self._item_idx.get(iid)

        bu = self._bu[u_idx] if u_idx is not None else 0.0
        bi = self._bi[i_idx] if i_idx is not None else 0.0
        dot = float(self._pu[u_idx] @ self._qi[i_idx]) if (u_idx is not None and i_idx is not None) else 0.0

        est = float(np.clip(gm + bu + bi + dot, 1.0, 5.0))
        return _Prediction(est)
