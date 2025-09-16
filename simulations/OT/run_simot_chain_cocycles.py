import torch
from run_cocycles_ot import run


def main():
    # Experiment settings
    n = 500
    ntrial = 20
    corrs = [0.1, 0.3, 0.5, 0.7, 0.9]
    additive = True
    multivariate = True
    learn_rate = 1e-2

    results = []

    for corr in corrs:
        for seed in range(ntrial):
            print(f"Running: corr={corr}, seed={seed}, additive={additive}, multivariate={multivariate}")
            res = run(
                n=n,
                seed=seed,
                corr=corr,
                additive=additive,
                multivariate_noise=multivariate,
                learn_rate=learn_rate,
            )
            results.append(res)

    # Save results to disk
    outfile = "cocycle_results_chain.pt"
    torch.save(results, outfile)
    print(f"Saved results to {outfile}")


if __name__ == "__main__":
    main()
