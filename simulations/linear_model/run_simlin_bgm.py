# Imports
import torch
from run_bgm import run_experiment
from csuite import SCMS

def main():
    # Experiment settings
    seeds = 50
    nsamples = 1000
    corr = 0.0
    use_dag = False
    scms = list(SCMS.keys())[:1]
    noises = ["normal", "rademacher", "cauchy", "gamma", "inversegamma"]
    learn_flow = True

    results = []

    for scm in scms:
        for noise in noises:
            for seed in range(seeds):
                print(f"Running: scm={scm}, noise={noise}, seed={seed}")
                res = run_experiment(
                    sc_name=scm,
                    noise_dist=noise,
                    seed=seed,
                    use_dag=use_dag,
                    corr=corr,
                    N=nsamples,
                    learn_flow=learn_flow,
                )
                results.append(res)

    # Save results to disk
    outfile = "bgm_linear_results.pt"
    torch.save(results, outfile)
    print(f"Saved results to {outfile}")


if __name__ == "__main__":
    main()
