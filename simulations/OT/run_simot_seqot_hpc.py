# Imports
from dask_jobqueue import SLURMCluster
from distributed import Client
from run_seqot import run
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
    corrs = [0.1, 0.3, 0.5, 0.7, 0.9]
    dist = "laplace"
    futures = []
    metadata = []

    for corr in corrs:
        for seed in range(ntrial):
            f0 = client.submit(
                run,
                seed,
                n,
                m,
                corr=corr,
                additive=False,
                multivariate_noise=False,
                dist=dist,
                wrongorder=False,
            )
            futures.append(f0)
            metadata.append(("seqot", "design_ii", False, corr, seed))

            f1 = client.submit(
                run,
                seed,
                n,
                m,
                corr=corr,
                additive=False,
                multivariate_noise=False,
                dist=dist,
                wrongorder=True,
            )
            futures.append(f1)
            metadata.append(("seqot", "design_ii", True, corr, seed))

    gathered = client.gather(futures)
    results = [meta + (result,) for meta, result in zip(metadata, gathered)]

    # Closing client
    client.close()
    cluster.close()

    torch.save(f="seqot_results.pt", obj=results)


if __name__ == "__main__":
    main()
