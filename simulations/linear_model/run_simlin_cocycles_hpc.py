# Imports
import torch
from dask_jobqueue import SLURMCluster
from distributed import Client
from run_cocycles import run_experiment
from csuite import SCMS

def main():

    seeds = 50
    nsamples = 1000
    corr = 0.0
    use_dag = False
    scms = list(SCMS.keys())[:1]
    noises = ["normal", "rademacher", "cauchy", "gamma", "inversegamma"]
    learn_flow = True
    
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
    for scm in scms:
        for noise in noises:
            for seed in range(seeds):
                f = client.submit(run_experiment, 
                                  sc_name = scm,
                                  noise_dist = noise,
                                  seed = seed,
                                  use_dag = use_dag,
                                  corr = corr,
                                  N = nsamples,
                                  learn_flow = learn_flow)
                futures += [f] 
    
    results = client.gather(futures)
    
    client.close()
    cluster.close()
    
    torch.save(f = "cocycles_linear_results.pt", obj = results)

if __name__ == "__main__":
    main()
