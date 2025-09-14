# Imports
import torch
from run_linear_model import run_experiment

def main():
    # Experiment settings
    seeds = 50
    nsamples = [1000]
    noise_types = ["normal", "gamma", "cauchy", "inversegamma", "rademacher"]

    results = []

    for seed in range(seeds):
        for n in nsamples:
            for noise in noise_types:
                print(f"Running: seed={seed}, n={n}, noise={noise}")
                res = run_experiment(seed, n, noise_type=noise)
                results.append(res)

    # Save results to disk
    outfile = "linear_model_results.pt"
    torch.save(results, outfile)
    print(f"Saved results to {outfile}")


if __name__ == "__main__":
    main()
