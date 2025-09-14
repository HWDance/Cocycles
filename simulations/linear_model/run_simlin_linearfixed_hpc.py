# Imports
import torch
from dask_jobqueue import SLURMCluster
from distributed import Client
from run_linear_model import run_experiment

def main():

    seeds = 50
    nsamples = [1000]
    
    cluster = SLURMCluster(
        n_workers=0,
        memory="8GB",
        processes=1,
        cores=1,
        scheduler_options={
            "dashboard_address": ":10092",
            "allowed_failures": 10
        },
        job_cpu=1,
        walltime="24:0:0",
    )
    cluster.adapt(minimum=0, maximum=200)
    client = Client(cluster)
    
    futures = []
    noise_types = ["normal", "gamma", "cauchy", "inversegamma", "rademacher"]
    
    for seed in range(seeds):
        for n in nsamples:
            for noise in noise_types:
                f = client.submit(run_experiment, seed, n, noise_type=noise)
                futures.append(f)
    
    results = client.gather(futures)
    
    client.close()
    cluster.close()
    
    torch.save(f = "linear_model_results.pt", obj = results)

if __name__ == "__main__":
    main()
