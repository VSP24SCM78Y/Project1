import numpy as np

class LassoHomotopyModel:
    def __init__(self):
        self.coef_path_ = []      # Stores beta vectors at each iteration
        self.lambda_path_ = []    # Stores lambda (max correlation) at each step

    def fit(self, X, y):
        n_samples, n_features = X.shape
        max_iter = min(n_samples - 1, n_features)

        # Center the data
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X = X - X_mean
        y = y - y_mean

        beta = np.zeros(n_features)
        residual = y.copy()
        active_set = []
        inactive_set = list(range(n_features))
        tol = 1e-6

        for _ in range(max_iter):
            corr = X.T @ residual
            c = np.max(np.abs(corr))

            # Store current state
            self.lambda_path_.append(c)
            self.coef_path_.append(beta.copy())

            if c < tol:
                break

            j = np.argmax(np.abs(corr))
            if j not in active_set:
                active_set.append(j)
                inactive_set.remove(j)

            X_active = X[:, active_set]
            try:
                G = X_active.T @ X_active
                G_inv = np.linalg.inv(G)
                A = 1 / np.sqrt(np.sum(G_inv))
                w = A * G_inv @ np.sign(corr[active_set])
                u = X_active @ w
                a = X.T @ u

                # Calculate gamma values for all inactive features
                gamma_candidates = []
                for i in inactive_set:
                    if abs(a[i]) > 1e-10:
                        denom1 = A - a[i]
                        denom2 = A + a[i]

                        if abs(denom1) > 1e-8:
                            gamma1 = (c - corr[i]) / denom1
                            if gamma1 > tol:
                                gamma_candidates.append(gamma1)

                        if abs(denom2) > 1e-8:
                            gamma2 = (c + corr[i]) / denom2
                            if gamma2 > tol:
                                gamma_candidates.append(gamma2)

                gamma = min(gamma_candidates) if gamma_candidates else c / A

                # Update coefficients for active features
                for idx, coef_idx in enumerate(active_set):
                    beta[coef_idx] += gamma * w[idx]

                residual = y - X @ beta
            except np.linalg.LinAlgError:
                print("Matrix inversion failed, stopping early.")
                break

        return LassoHomotopyResults(beta, X_mean, y_mean)


class LassoHomotopyResults:
    def __init__(self, coef, X_mean, y_mean):
        self.coef_ = coef
        self.X_mean = X_mean
        self.y_mean = y_mean

    def predict(self, x):
        return (x - self.X_mean) @ self.coef_ + self.y_mean
