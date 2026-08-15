"""Combined OT + seqOT launcher for design (i).

Use this file when you want the historical paired OT vs seqOT benchmark.
For seqOT-only wrong-order sweeps, use run_simot_chain_seqot_hpc.py instead.
"""

import torch
from run_seqot_chain import run as run_seqot
from run_ot import run as run_ot


def main():
    print("Running combined OT + seqOT launcher for design (i). For seqOT wrong-order sweeps, use run_simot_chain_seqot_hpc.py.")
    # Experiment settings
    n = 500
    m = n
    ntrial = 20
    additive = True
    corrs = [0.1, 0.3, 0.5, 0.7, 0.9]
    multivariate = True
    dist = "laplace"

    results = []

    for corr in corrs:
        for seed in range(ntrial):
            print(f"Running seqOT: corr={corr}, seed={seed}")
            res_seqot = run_seqot(
                seed,
                n,
                m,
                corr=corr,
                additive=additive,
                multivariate_noise=multivariate,
                dist=dist,
            )
            results.append(("seqot", corr, seed, res_seqot))

            print(f"Running OT: corr={corr}, seed={seed}")
            res_ot = run_ot(
                seed,
                n,
                m,
                "sqeuclidean",
                corr=corr,
                additive=additive,
                multivariate_noise=multivariate,
                dist=dist,
            )
            results.append(("ot", corr, seed, res_ot))

    outfile = "OT_results_chain.pt"
    torch.save(results, outfile)
    print(f"Saved results to {outfile}")


if __name__ == "__main__":
    main()
