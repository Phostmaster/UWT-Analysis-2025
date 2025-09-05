# Install dependencies (run this in a Jupyter cell or terminal first)
# !pip install numpy pandas matplotlib scipy scikit-learn

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import KFold
from scipy.interpolate import UnivariateSpline
from scipy.stats import probplot
from sklearn.linear_model import Ridge

# Helpers
G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
MPC_M = 3.085677581e22  # m
MSUN_KG = 1.98847e30  # kg

def rho_crit_SI(H0_km_s_Mpc: float) -> float:
    """Critical density in kg/m^3 from H0 (km/s/Mpc)."""
    H0_SI = (H0_km_s_Mpc * 1000.0) / MPC_M  # s^-1
    return 3.0 * H0_SI**2 / (8.0 * math.pi * G_SI)

def Msun_per_Mpc3_to_SI(rho: float) -> float:
    """Convert M☉/Mpc^3 to kg/m^3."""
    return rho * MSUN_KG / (MPC_M**3)

def safe_log(x):
    return np.log(np.clip(np.asarray(x, dtype=float), 1e-300, None))

def fit_linear(x, y) -> Tuple[float, float, float]:
    """Simple OLS y = a + b x; returns (a, b, R^2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return beta[0], beta[1], r2

def fit_multi_linear(x, y, z, alpha=0.005) -> Tuple[float, float, float, float]:
    """Multi-linear fit Y = a + b*X + c*z; returns (a, b, c, R2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 3:
        return [np.nan] * 4
    M = np.vstack([np.ones_like(x), x, z]).T
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(M, y)
    beta = ridge.coef_
    yhat = M @ beta
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return beta[0], beta[1], beta[2], r2

def fit_polynomial(x, y, degree=2) -> Tuple[float, float, float, float]:
    """Polynomial fit y = a + b*x + c*x^2 + ...; returns (a, b, c, ..., R^2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < degree + 1:
        return [np.nan] * (degree + 2)
    try:
        coeffs = np.polyfit(x, y, degree)
        yhat = np.polyval(coeffs, x)
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        return coeffs[degree], coeffs[degree-1], coeffs[degree-2], r2
    except (np.linalg.LinAlgError):
        return [np.nan] * (degree + 2)

def fit_spline(x, y, df_range=(3, 4)) -> Tuple[float, float, float, float]:
    """Fit a spline with 3–4 degrees of freedom and compute R2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return np.nan, np.nan, np.nan, np.nan
    best_r2, best_spline = -np.inf, None
    for df in range(df_range[0], df_range[1] + 1):
        spline = UnivariateSpline(x, y, k=df, s=0.1)  # Slight smoothing for stability
        yhat = spline(x)
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        if r2 > best_r2 and not np.isnan(r2):
            best_r2, best_spline = r2, spline
    if best_spline is None:
        return np.nan, np.nan, np.nan, np.nan
    yhat = best_spline(x)
    return np.nan, np.nan, np.nan, best_r2  # Spline coeffs not directly comparable

def fit_best_polynomial_cv(x, y, max_degree=4, cv_folds=5, min_r2_improvement=0.02) -> Tuple[float, float, float, float, float]:
    """Fit polynomial or spline up to max_degree with cross-validation to select best model based on adjusted R^2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < max_degree + 1:
        return [np.nan] * 5
    best_r2_adj, best_coeffs, best_degree, best_aic = -np.inf, [], 0, np.inf
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    n = len(x)
    prev_r2_adj = -np.inf
    # Polynomial fits
    for degree in range(1, max_degree + 1):
        r2_scores = []
        for train_idx, test_idx in kf.split(x):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            try:
                coeffs = np.polyfit(x_train, y_train, degree)
                yhat = np.polyval(coeffs, x_test)
                ss_res = np.sum((y_test - yhat)**2)
                ss_tot = np.sum((y_test - np.mean(y_test))**2)
                r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
                if not np.isnan(r2):
                    r2_adj = 1 - (1 - r2) * (n - 1) / (n - degree - 1 - (degree - 1) * 0.15)  # Stricter penalty
                    r2_scores.append(r2_adj)
            except (np.linalg.LinAlgError):
                continue
        mean_r2_adj = np.mean(r2_scores) if r2_scores else np.nan
        try:
            coeffs = np.polyfit(x, y, degree)
            yhat = np.polyval(coeffs, x)
            aic = aic_from_residuals(y, yhat, degree + 1)[0]
            if mean_r2_adj > best_r2_adj and mean_r2_adj - prev_r2_adj >= min_r2_improvement and not np.isnan(mean_r2_adj):
                best_r2_adj, best_coeffs, best_degree, best_aic = mean_r2_adj, coeffs, degree, aic
        except (np.linalg.LinAlgError):
            continue
        prev_r2_adj = mean_r2_adj
        print(f"Degree {degree}: Mean CV Adjusted R² = {mean_r2_adj:.6f}, AIC = {aic:.6f}")
    # Spline fits (3–4 df) only if x is strictly increasing
    if np.all(np.diff(x) > 0):
        for df in range(3, 5):
            r2_scores = []
            for train_idx, test_idx in kf.split(x):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                spline = UnivariateSpline(x_train, y_train, k=df, s=0.1)  # Slight smoothing for stability
                yhat = spline(x_test)
                ss_res = np.sum((y_test - yhat)**2)
                ss_tot = np.sum((y_test - np.mean(y_test))**2)
                r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
                if not np.isnan(r2):
                    r2_adj = 1 - (1 - r2) * (n - 1) / (n - df - 1 - (df - 1) * 0.15)
                    r2_scores.append(r2_adj)
            mean_r2_adj = np.mean(r2_scores) if r2_scores else np.nan
            spline = UnivariateSpline(x, y, k=df, s=0.1)
            yhat = spline(x)
            aic, bic = aic_from_residuals(y, yhat, df + 1)  # Approximate k_params as df + 1
            if mean_r2_adj > best_r2_adj and mean_r2_adj - prev_r2_adj >= min_r2_improvement and not np.isnan(mean_r2_adj):
                best_r2_adj, best_coeffs, best_degree, best_aic = mean_r2_adj, [np.nan, np.nan, np.nan, spline], -df, aic
            prev_r2_adj = mean_r2_adj
            print(f"DF {df}: Mean CV Adjusted R² = {mean_r2_adj:.6f}, AIC = {aic:.6f}")
    print(f"Selected degree: {best_degree} with Adjusted R² = {best_r2_adj:.6f}, AIC = {best_aic:.6f}")
    if best_degree > 0:
        return best_coeffs[best_degree], best_coeffs[best_degree-1], best_coeffs[best_degree-2] if best_degree > 1 else np.nan, best_r2_adj, best_aic
    else:
        return np.nan, np.nan, np.nan, best_r2_adj, best_aic  # Spline case

def aic_from_residuals(y, yhat, k_params: int) -> Tuple[float, float]:
    n = len(y)
    if n < 2:
        return np.nan, np.nan
    sse = np.sum((y - yhat)**2)
    sigma2 = sse / n if n > 0 else np.nan
    if sigma2 <= 0 or n == 0:
        return np.nan, np.nan
    aic = n * (1 + np.log(2*math.pi*sigma2)) + 2 * k_params
    bic = aic + math.log(n) * (k_params - 2)
    return aic, bic

def bootstrap_ci(x, y, log_mode=False, B=2000, seed=42):
    """Bootstrap 95% CI for slope (and intercept) in linear or log-log space."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return (np.nan, np.nan, np.nan), (np.nan,)
    idx = np.arange(len(x))
    slopes = []
    intercepts = []
    for _ in range(B):
        ii = rng.choice(idx, size=len(idx), replace=True)
        xb, yb = x[ii], y[ii]
        if log_mode:
            xb = safe_log(xb)
            yb = safe_log(yb)
        a, b, _ = fit_linear(xb, yb)
        if not np.isnan(b):
            intercepts.append(a)
            slopes.append(b)
    if not slopes:
        return (np.nan, np.nan, np.nan), (np.nan,)
    lo = np.percentile(slopes, 2.5)
    hi = np.percentile(slopes, 97.5)
    return (float(np.mean(slopes)), float(lo), float(hi)), (float(np.mean(intercepts)),)

# Load data
csv_path = r'C:\Users\admin\Desktop\private\UWT_EP_Collab\uwt_observables.csv'
try:
    df = pd.read_csv(csv_path, encoding='utf-8', sep='\t')  # Added sep='\t' to handle tab-separated CSV
except FileNotFoundError:
    print(f"Error: {csv_path} not found. Please check the file path.")
    raise SystemExit

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Debug: Print column names
print("CSV columns:", list(df.columns))

# Fill missing H0 with 70
if "H0_km_s_Mpc" not in df.columns:
    df["H0_km_s_Mpc"] = 70.0
df["H0_km_s_Mpc"] = df["H0_km_s_Mpc"].fillna(70.0)

# Global Φ, ε if missing per-row
if "Phi1_GeV" not in df.columns:
    df["Phi1_GeV"] = 0.226
if "Phi2_GeV" not in df.columns:
    df["Phi2_GeV"] = 0.094

# Check for required columns
required_cols = ["rho_star_Msun_per_Mpc3", "rho_bh_Msun_per_Mpc3"]
if not any(col in df.columns for col in required_cols):
    print(f"Error: At least one of {required_cols} is required in {csv_path}. Found columns: {list(df.columns)}")
    raise SystemExit

# Dimensionless groups
df["rho_c_SI"] = df["H0_km_s_Mpc"].apply(rho_crit_SI)
for col in ["rho_star_Msun_per_Mpc3", "rho_bh_Msun_per_Mpc3"]:
    if col in df.columns:
        si = df[col].astype(float).apply(Msun_per_Mpc3_to_SI)
        df[col.replace("Msun_per_Mpc3", "SI")] = si
        df[col.replace("Msun_per_Mpc3", "hat")] = si / df["rho_c_SI"]
df["Phi_prod_GeV2"] = (df["Phi1_GeV"].astype(float) * df["Phi2_GeV"].astype(float)).abs()
df["X_hat"] = df["Phi_prod_GeV2"]
df["X_hat_inv"] = 1 / (df["Phi_prod_GeV2"] + 1e-10)

# Targets to fit
targets = []
if "rho_star_hat" in df.columns:
    targets.append(("rho_star_hat", "ρ_star / ρ_c", "X_hat_inv"))
if "rho_bh_hat" in df.columns:
    targets.append(("rho_bh_hat", "ρ_BH / ρ_c", "X_hat"))
if not targets:
    print(f"Error: No valid target columns (rho_star_hat or rho_bh_hat) found in {csv_path}.")
    raise SystemExit

output_dir = r'C:\Users\admin\Desktop\uwt_dimensional_outputs'
Path(output_dir).mkdir(exist_ok=True)
summary_rows = []
for tgt_col, tgt_label, x_col in targets:
    # Build regressors
    X = df[x_col].values
    Y = df[tgt_col].values
    mask = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[mask], Y[mask]
    z = df["z"].values[mask]
    # Multi-linear: Y = a + b*X + c*z
    a_multi, b_multi, c_multi, r2_multi = fit_multi_linear(X, Y, z, alpha=0.005)
    Yhat_multi = a_multi + b_multi*X + c_multi*z if not np.isnan(a_multi) else np.zeros_like(Y)
    aic_multi, bic_multi = aic_from_residuals(Y, Yhat_multi, k_params=3)
    # Linear: Y = a + b X
    a, b, r2 = fit_linear(X, Y)
    Yhat = a + b*X if not np.isnan(a) else np.zeros_like(Y)
    aic, bic = aic_from_residuals(Y, Yhat, k_params=2)
    (b_mean, b_lo, b_hi), _ = bootstrap_ci(X, Y, log_mode=False)
    # Best polynomial or spline fit (up to degree 4 with adjusted R^2 and min improvement)
    a_poly, b_poly, c_poly, r2_poly, aic_poly = fit_best_polynomial_cv(X, Y, max_degree=4)
    if isinstance(a_poly, UnivariateSpline):
        Yhat_poly = a_poly(X)
    else:
        Yhat_poly = np.polyval([a_poly, b_poly, c_poly] + [0] * (4 - len([a_poly, b_poly, c_poly])), X)
    aic_poly, bic_poly = aic_from_residuals(Y, Yhat_poly, k_params=len([a_poly, b_poly, c_poly]) + 1 if not isinstance(a_poly, UnivariateSpline) else 4)
    # Log-log: log Y vs log X
    Xlog = safe_log(X)
    Ylog = safe_log(Y)
    aL, bL, r2L = fit_linear(Xlog, Ylog)
    Ylog_hat = aL + bL*Xlog if not np.isnan(aL) else np.zeros_like(Ylog)
    aicL, bicL = aic_from_residuals(Ylog, Ylog_hat, k_params=2)
    (bL_mean, bL_lo, bL_hi), _ = bootstrap_ci(X, Y, log_mode=True)
    # Residuals
    residuals = Y - Yhat_multi if not np.isnan(a_multi) else Y - Yhat
    res_mean = np.mean(residuals) if len(residuals) > 0 else np.nan
    res_std = np.std(residuals) if len(residuals) > 0 else np.nan
    # Save plots with color by z
    plt.figure()
    plt.scatter(X, Y, c=df["z"][mask], cmap='viridis', s=16, label="Data")
    xs = np.linspace(X.min(), X.max(), 200) if len(X) > 1 else X
    plt.plot(xs, a + b*xs, label="Linear Fit")
    if isinstance(a_poly, UnivariateSpline):
        plt.plot(xs, a_poly(xs), label=f"Spline Fit (df={-best_degree})", linestyle='--')
    else:
        plt.plot(xs, np.polyval([a_poly, b_poly, c_poly] + [0] * (4 - len([a_poly, b_poly, c_poly])), xs), label=f"Poly Fit (deg={len([a_poly, b_poly, c_poly]) - 1})", linestyle='--')
    plt.plot(xs, a_multi + b_multi*xs + c_multi*df["z"].values.mean(), label="Multi-Linear Fit", linestyle='-.')
    plt.xlabel("1/|Φ₁Φ₂| (GeV⁻²)" if x_col == "X_hat_inv" else "|Φ₁Φ₂| (GeV²)")
    plt.ylabel(tgt_label)
    plt.title(f"Fit: {tgt_label} vs {'1/|Φ₁Φ₂|' if x_col == 'X_hat_inv' else '|Φ₁Φ₂|'}")
    plt.legend()
    plt.colorbar(label="Redshift (z)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{tgt_col}_linear.png")
    plt.close()
    # Residuals plot
    plt.figure()
    plt.scatter(X, residuals, c=df["z"][mask], cmap='viridis', label="Residuals")
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("1/|Φ₁Φ₂| (GeV⁻²)" if x_col == "X_hat_inv" else "|Φ₁Φ₂| (GeV²)")
    plt.ylabel("Residuals")
    plt.title(f"Residuals: {tgt_label} vs {'1/|Φ₁Φ₂|' if x_col == 'X_hat_inv' else '|Φ₁Φ₂|'}")
    plt.colorbar(label="Redshift (z)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{tgt_col}_residuals.png")
    plt.close()
    # QQ plot
    plt.figure()
    probplot(residuals, dist="norm", plot=plt)
    plt.title(f"QQ Plot: {tgt_label} Residuals")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{tgt_col}_qqplot.png")
    plt.close()
    plt.figure()
    plt.scatter(Xlog, Ylog, c=df["z"][mask], cmap='viridis', s=16, label="Data")
    xsL = np.linspace(Xlog.min(), Xlog.max(), 200) if len(Xlog) > 1 else Xlog
    plt.plot(xsL, aL + bL*xsL, label="Log-Log Fit")
    plt.xlabel(f"log {'1/|Φ₁Φ₂|' if x_col == 'X_hat_inv' else '|Φ₁Φ₂|'}")
    plt.ylabel(f"log {tgt_label}")
    plt.title(f"Log–log fit: {tgt_label} vs {'log 1/|Φ₁Φ₂|' if x_col == 'X_hat_inv' else 'log |Φ₁Φ₂|'}")
    plt.legend()
    plt.colorbar(label="Redshift (z)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{tgt_col}_loglog.png")
    plt.close()
    # Collect summary
    summary_rows.append({
        "target": tgt_col,
        "predictor": x_col,
        "N": int(len(X)),
        "linear_intercept_a": a,
        "linear_slope_b": b,
        "linear_slope_b_boot_mean": b_mean,
        "linear_slope_b_CI95_lo": b_lo,
        "linear_slope_b_CI95_hi": b_hi,
        "linear_R2": r2,
        "linear_AIC": aic,
        "linear_BIC": bic,
        "multi_R2": r2_multi,
        "multi_AIC": aic_multi,
        "multi_BIC": bic_multi,
        "poly_R2": r2_poly,
        "poly_AIC": aic_poly,
        "poly_BIC": bic_poly,
        "loglog_intercept_a": aL,
        "loglog_slope_b": bL,
        "loglog_slope_b_boot_mean": bL_mean,
        "loglog_slope_b_CI95_lo": bL_lo,
        "loglog_slope_b_CI95_hi": bL_hi,
        "loglog_R2": r2L,
        "loglog_AIC": aicL,
        "loglog_BIC": bicL,
        "residual_mean": res_mean,
        "residual_std": res_std
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(f"{output_dir}/fit_summary.csv", index=False)
print("\n=== Fit summary ===")
print(summary.to_string(index=False))
print("\nSaved:")
print(f" - {output_dir}/fit_summary.csv")
print(f" - {output_dir}/*_linear.png")
print(f" - {output_dir}/*_residuals.png")
print(f" - {output_dir}/*_qqplot.png")
print(f" - {output_dir}/*_loglog.png")
