import torch
from run_flows_mixed import run_experiment
from csuite_mixed import SCMS


def main():
    # Experiment settings
    seeds = 10
    nsamples = 2000
    corr = 0.0
    use_dag = False
    affine = True
    consistent = False
    scms = list(SCMS.keys())

    results = []

    for scm in scms:
        for seed in range(seeds):
            print(f"Running: scm={scm}, seed={seed}, affine={affine}, consistent={consistent}")
            res = run_experiment(
                sc_name=scm,
                seed=seed,
                use_dag=use_dag,
                corr=corr,
                N=nsamples,
                affine=affine,
                consistent=consistent,
            )
            results.append(res)

    # Save results to disk
    outfile = "causalflow_csuite_results.pt"
    torch.save(results, outfile)
    print(f"Saved results to {outfile}")


if __name__ == "__main__":
    main()
