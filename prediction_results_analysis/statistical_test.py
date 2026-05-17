import numpy as np
import pandas as pd

K = 5   # folds
S = 12   # seeds

def compute_diffs(df: pd.DataFrame) -> np.ndarray:
    """
    Returns diffs array of shape (K, S): d[f, s] = aucpr_A - aucpr_B.
    Expects columns: fold, seed, model, aucpr.
    """
    pivot = df.pivot_table(index=["fold", "seed"], columns="model", values="aucpr", aggfunc="mean")

    # Sanity checks
    if not {"A", "B"}.issubset(set(pivot.columns)):
        raise ValueError(f"Expected models A and B in column 'model'. Found: {list(pivot.columns)}")

    pivot = pivot.reset_index()

    # Ensure we have full grid (fold x seed)
    folds = sorted(pivot["fold"].unique())
    seeds = sorted(pivot["seed"].unique())
    if len(folds) != K or len(seeds) != S:
        raise ValueError(f"Expected {K} folds and {S} seeds, got {len(folds)} folds and {len(seeds)} seeds.")

    # Build (K,S) diffs
    diffs = np.empty((K, S), dtype=float)
    fold_to_i = {f: i for i, f in enumerate(folds)}
    seed_to_j = {s: j for j, s in enumerate(seeds)}

    for _, row in pivot.iterrows():
        i = fold_to_i[row["fold"]]
        j = seed_to_j[row["seed"]]
        diffs[i, j] = float(row["A"] - row["B"])

    if np.isnan(diffs).any():
        raise ValueError("Found NaNs in diffs. Likely missing A/B for some (fold, seed).")

    return diffs


def effect_D(diffs: np.ndarray) -> float:
    """Equal weight for folds: D = mean_f( mean_s( d[f,s] ) )."""
    return diffs.mean(axis=1).mean(axis=0)


def signflip_permutation_test(diffs: np.ndarray, n_perm: int = 200_000,
                              alternative: str = "two-sided", rng_seed: int = 0):
    """
    Paired + blocked sign-flip permutation test using statistic D.
    alternative: "two-sided" | "greater" | "less"
      - "greater": H1: A > B  (D > 0)
      - "less":    H1: A < B
    """
    rng = np.random.default_rng(rng_seed)
    D_obs = effect_D(diffs)

    # Random sign flips: shape (n_perm, K, S)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diffs.shape[0], diffs.shape[1]))
    D_perm = (signs * diffs).mean(axis=2).mean(axis=1)  # mean over seeds, then folds

    if alternative == "greater":
        p = (np.sum(D_perm >= D_obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(D_perm <= D_obs) + 1) / (n_perm + 1)
    elif alternative == "two-sided":
        p = (np.sum(np.abs(D_perm) >= abs(D_obs)) + 1) / (n_perm + 1)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return D_obs, p


def hierarchical_bootstrap_ci(diffs: np.ndarray, n_boot: int = 50_000,
                              ci: float = 0.95, rng_seed: int = 0):
    """
    Hierarchical bootstrap: resample folds, then within each fold resample seeds.
    Returns (lo, hi).
    """
    rng = np.random.default_rng(rng_seed)
    K, S = diffs.shape
    boot = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        fold_idx = rng.integers(0, K, size=K)  # resample folds
        fold_means = np.empty(K, dtype=float)

        for t, fi in enumerate(fold_idx):
            seed_idx = rng.integers(0, S, size=S)  # resample seeds within fold
            fold_means[t] = diffs[fi, seed_idx].mean()

        boot[b] = fold_means.mean()

    alpha = 1.0 - ci
    lo = np.quantile(boot, alpha / 2)
    hi = np.quantile(boot, 1 - alpha / 2)
    return lo, hi


if __name__ == "__main__":
    # Example:
    df = pd.read_csv("results.csv")  # must contain fold, seed, model, aucpr
    diffs = compute_diffs(df)

    D_obs, p = signflip_permutation_test(diffs, alternative="two-sided")
    lo, hi = hierarchical_bootstrap_ci(diffs)

    print(f"D (mean AUCPR diff A-B) = {D_obs:.6f}")
    print(f"95% CI: [{lo:.6f}, {hi:.6f}]")
    print(f"p-value: {p:.6g}")
