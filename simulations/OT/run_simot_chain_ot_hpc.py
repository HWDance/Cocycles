# Imports
from dask_jobqueue import SLURMCluster
from distributed import Client
from run_seqot_chain import run as run_seqot
from run_ot import run as run_ot
import torch

def main():
    # Cluster creation
    cluster = SLURMCluster(
        n_workers=0,
        memory="32GB",
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
    m = n
    ntrial = 20
    additive = True
    corrs = [0.1,0.3,0.5,0.7,0.9]
    multivariate = True
    dist = "laplace"
    futures = []
    metadata = []
        
    for corr in corrs:
        for seed in range(ntrial):
            f1 = client.submit(
                run_seqot,
                seed,
                n,
                m,
                corr=corr,
                additive=additive,
                multivariate_noise=multivariate,
                dist=dist,
            )
            futures.append(f1)
            metadata.append(("seqot", corr, seed))

            f3 = client.submit(
                run_ot,
                seed,
                n,
                m,
                "sqeuclidean",
                corr=corr,
                additive=additive,
                multivariate_noise=multivariate,
                dist=dist,
            )
            futures.append(f3)
            metadata.append(("ot", corr, seed))
    
    gathered = client.gather(futures)
    results = [meta + (result,) for meta, result in zip(metadata, gathered)]
    
    # Closing client
    client.close()
    cluster.close()
    
    torch.save(f = "OT_results_chain.pt", obj = results)

    
if __name__ == "__main__":
    main()
