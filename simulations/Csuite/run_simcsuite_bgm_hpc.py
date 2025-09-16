# Imports
import torch
from dask_jobqueue import SLURMCluster
from distributed import Client
from run_bgm_mixed import run_experiment
from csuite_mixed import SCMS

seeds = 10
nsamples = 2000
corr = 0.0
use_dag = False
affine = False
consistent = False
scms = list(SCMS.keys())

def main():
    cluster = SLURMCluster(
        n_workers=0,
        memory="20GB",
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
    
    scms
    
    futures = []
    for scm in scms:
        for seed in range(seeds):
            f = client.submit(run_experiment, 
                              sc_name = scm,
                              seed = seed,
                              use_dag = use_dag,
                              corr = corr,
                              N = nsamples,
                              affine = affine,
                              consistent = consistent)
            futures += [f] 
    
    results = client.gather(futures)
    
    client.close()
    cluster.close()
    
    torch.save(f = "bgm_csuite_results.pt", obj = results)


if __name__ == "__main__":
    main()
