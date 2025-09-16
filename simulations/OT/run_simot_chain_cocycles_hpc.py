# Imports
import torch
from dask_jobqueue import SLURMCluster
from distributed import Client
from run_cocycles_ot import run

def main():
        
    # Cluster creation
    cluster = SLURMCluster(
        n_workers=0,
        memory="16GB",
        processes=1,
        cores=1,
        scheduler_options={
            "dashboard_address": ":10095",
            "allowed_failures": 10
        },
        job_cpu=1,
        walltime="24:0:0",
        #job_extra_directives = ["-p medium,fast,cpu"],
    )
    cluster.adapt(minimum=0, maximum=100)
    client = Client(cluster)
    
    # Submitting jobs
    n = 500
    ntrial = 20
    corrs = [0.1,0.3,0.5,0.7,0.9]
    additive = True
    multivariate = True
    futures = []
    learn_rate = 1e-2
    for corr in corrs:
        for seed in range(ntrial):
            f0 = client.submit(run,n = n, seed = seed, corr = corr, additive = additive, multivariate_noise = multivariate, learn_rate = learn_rate)
            futures += [f0]
    
    futures
    
    # Gettaing results
    results = client.gather(futures)
    
    # Closing client
    client.close()
    cluster.close()
    
    torch.save(f = "cocycle_results_chain", obj = results)

if __name__ == "__main__":
    main()
