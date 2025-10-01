# model
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def pls_run(X_train, y_train, X_test, y_test, n_components=4):
    """
    PLS regression that accepts 2D or higher inputs.
    - X_*: [N, F] or [N, ...]  -> auto-flatten to [N, F]
    - y_*: [N] or [N, T]       -> ensure 2D [N, T]
    """

    # ---- to numpy ----
    X_train = X_train.cpu().numpy() if hasattr(X_train, "cpu") else np.asarray(X_train)
    X_test  = X_test.cpu().numpy()  if hasattr(X_test,  "cpu") else np.asarray(X_test)
    y_train = y_train.cpu().numpy() if hasattr(y_train, "cpu") else np.asarray(y_train)
    y_test  = y_test.cpu().numpy()  if hasattr(y_test,  "cpu") else np.asarray(y_test)

    # ---- flatten X to 2D if needed ----
    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if X_test.ndim > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)

    # ---- ensure y is 2D ----
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)


    # ---- fit & predict ----
    pls = PLSRegression(n_components=n_components,scale=True)
    pls.fit(X_train, y_train)
    y_pred = pls.predict(X_test)

    # ---- metrics ----
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    # Pearson correlation per target, then mean
    corrs = []
    for t in range(y_test.shape[1]):
        a = y_test[:, t].ravel()
        b = y_pred[:, t].ravel()
        if np.std(a) > 0 and np.std(b) > 0:
            corrs.append(np.corrcoef(a, b)[0, 1])
        else:
            corrs.append(np.nan)
    corr = np.nanmean(corrs)

    print("PLS Results:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  MSE  = {mse:.4f}")
    print(f"  RMSE = {np.sqrt(mse):.4f}")
    print(f"  R²   = {r2:.4f}")
    print(f"  Corr = {corr:.4f}")

    return corr, y_test, y_pred