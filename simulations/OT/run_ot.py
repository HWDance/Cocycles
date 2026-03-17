import numpy as np
import ot
from dgp import generate_ot_data

# SCM map: Y(x) = mu(x) + A(x) @ xi
def affine_ot_map(m_src, S_src, m_tgt, S_tgt):
    L_src = np.linalg.cholesky(S_src)
    L_tgt = np.linalg.cholesky(S_tgt)
    A = L_tgt @ np.linalg.inv(L_src)
    b = m_tgt - A @ m_src
    return A, b

def run(seed, n=None, m=None, Dist="sqeuclidean", corr = 0.5, additive = True, multivariate_noise = False, dist = "laplace"):
    data = generate_ot_data(
        seed=seed,
        n=n,
        m=m,
        corr=corr,
        additive=additive,
        multivariate_noise=multivariate_noise,
        dist=dist,
    )
    n0, n1, n2 = data["n0"], data["n1"], data["n2"]
    P0, P1, P2 = data["P0"], data["P1"], data["P2"]
    Y1_cf, Y2_cf = data["Y1_cf"], data["Y2_cf"]

    # Uniform weights
    a0 = np.ones(n0) / n0
    a1 = np.ones(n1) / n1
    a2 = np.ones(n2) / n2

    # Cost matrices
    C01 = ot.dist(P0, P1, Dist)
    C02 = ot.dist(P0, P2, Dist)
    C12 = ot.dist(P1, P2, Dist)

    # Transport plans (exact OT)
    T01 = ot.emd(a0, a1, C01) / a0[:, None]
    T02 = ot.emd(a0, a2, C02) / a0[:, None]
    T12 = ot.emd(a1, a2, C12) / a1[:, None]

    # OT barycentric projections
    Tmap_01 = T01 @ P1
    Tmap_02 = T02 @ P2
    Tmap_12 = T12 @ P2

    # Permutation route for composite map
    perm = np.argmax(T01, axis=1)
    Tmap_012 = Tmap_12[perm]
    Tmap_01_permuted = P1[perm]

    # Estimated differences from OT maps
    diff_est_10 = Tmap_01 - P0
    diff_est_21_direct = Tmap_02 - Tmap_01
    diff_est_21_composite = Tmap_012 - Tmap_01
    diff_est_20_direct = Tmap_02 - P0
    diff_est_20_composite = Tmap_012 - P0

    # True counterfactual shifts
    true_diff_10 = Y1_cf - P0
    true_diff_21 = Y2_cf - Y1_cf
    true_diff_20 = Y2_cf - P0

    # ------------------------------
    # Errors
    error_10 = diff_est_10 - true_diff_10
    error_21_direct = diff_est_21_direct - true_diff_21
    error_21_composite = diff_est_21_composite - true_diff_21
    error_20_direct = diff_est_20_direct - true_diff_20
    error_20_composite = diff_est_20_composite - true_diff_20
    norm_inconsistency = np.linalg.norm(Tmap_02 - Tmap_012, axis=1)

    # Norms
    norm_error_10 = np.linalg.norm(error_10, axis=1)
    norm_error_21_direct = np.linalg.norm(error_21_direct, axis=1)
    norm_error_21_composite = np.linalg.norm(error_21_composite, axis=1)
    norm_error_20_direct = np.linalg.norm(error_20_direct, axis=1)
    norm_error_20_composite = np.linalg.norm(error_20_composite, axis=1)

    # ------------------------------
    return {
        "seed": seed,
        "name": f"OT_dist={Dist}",
        "corr": corr,
        "additive": additive,
        "RMSE10": norm_error_10.mean(),
        "RMSE21direct": norm_error_21_direct.mean(),
        "RMSE21composite": norm_error_21_composite.mean(),
        "RMSE20direct": norm_error_20_direct.mean(),
        "RMSE20composite": norm_error_20_composite.mean(),
        "RMSEinconsistency": norm_inconsistency.mean(),
        "ATE10": np.sqrt((error_10.mean(0) ** 2).mean()),
        "ATE21direct": np.sqrt((error_21_direct.mean(0) ** 2).mean()),
        "ATE21composite": np.sqrt((error_21_composite.mean(0) ** 2).mean()),
        "ATE20direct": np.sqrt((error_20_direct.mean(0) ** 2).mean()),
        "ATE20composite": np.sqrt((error_20_composite.mean(0) ** 2).mean()),
    }

if __name__ == "__main__":
    result = run(seed=0, n=50, corr=0.5, additive=False, multivariate_noise=False)
    print("\n===== OT vs SCM Test Results =====")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k:>20}: {v:.6f}")
        else:
            print(f"{k:>20}: {v}")
