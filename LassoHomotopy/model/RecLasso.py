import numpy as np
from copy import deepcopy
from .LassoHomotopy import LassoHomotopyModel


class RecLassoModel:
    """
    RecLasso: Homotopy-based Online LASSO.
    Efficiently updates LASSO coefficients when new data points arrive.
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.beta = None
        self.X_mean = None
        self.y_mean = None
        self.active_set = []
        self.G_inv = None

    def fit(self, X, y):
        """
        Initial training using LASSO Homotopy.
        Stores active set and precomputes Gram matrix inverse.
        """
        self.X = deepcopy(X)
        self.y = deepcopy(y)

        self.X_mean = np.mean(X, axis=0)
        self.y_mean = np.mean(y)
        X_centered = X - self.X_mean
        y_centered = y - self.y_mean

        model = LassoHomotopyModel()
        results = model.fit(X_centered, y_centered)
        self.beta = results.coef_

        self.active_set = list(np.flatnonzero(self.beta))
        if self.active_set:
            X_A = X_centered[:, self.active_set]
            G = X_A.T @ X_A
            self.G_inv = np.linalg.inv(G)
        else:
            self.G_inv = None

        return self

    def partial_fit(self, x_new, y_new):
        if self.beta is None:
            raise RuntimeError("Model must be fitted first.")

        # Center new data
        x_new = x_new - self.X_mean
        y_new = y_new - self.y_mean

        A = self.active_set
        v = np.sign(self.beta[A])
        x_A = x_new[A]

        if len(A) == 0:
            print("Active set is empty. Skipping update.")
            return

        # Compute new inverse via rank-1 update
        X_A = self.X[:, A]
        X_aug = np.vstack([X_A, x_A])
        y_aug = np.append(self.y, y_new)

        try:
            G = X_aug.T @ X_aug
            G_inv_new = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            print("Rank-deficient matrix in update.")
            return

        theta_0 = self.beta[A]
        theta_1 = G_inv_new @ (X_aug.T @ y_aug - v)
        w = theta_1 - theta_0

        # Transition loop (homotopy from t=0 to t=1)
        t = 0.0
        t_final = 1.0
        tol = 1e-6

        while t < t_final:
            gamma_candidates = [1.0]  # default final step
            # Case 1: coefficient becomes zero
            for i, b in enumerate(theta_0):
                if abs(w[i]) > tol:
                    gamma_i = -b / w[i]
                    if tol < gamma_i < 1.0:
                        gamma_candidates.append(gamma_i)
            # Choose smallest gamma to next transition
            gamma = min(gamma_candidates)
            t += gamma

            # Update beta
            theta = theta_0 + gamma * w
            for idx, j in enumerate(A):
                self.beta[j] = theta[idx]

            # Check for Case 1: coefficient hits 0
            for i, b in enumerate(theta):
                if abs(b) < tol:
                    j = A[i]
                    print(f"Removing feature {j} from active set")
                    self.active_set.remove(j)
                    self.beta[j] = 0

            break  # Stop at first transition — simplified loop

        # Update data and Gram inverse
        self.X = np.vstack([self.X, x_new])
        self.y = np.append(self.y, y_new)
        self.G_inv = G_inv_new
