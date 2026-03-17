try:
    from .run_backdoor_seqot import run as run_seqot
except ImportError:
    from run_backdoor_seqot import run as run_seqot


def run(
    seed=0,
    n=500,
    rho=0.5,
    logitscale=1.0,
    condition_on_z=False,
    epsilon=0.0,
):
    result = run_seqot(
        seed=seed,
        n=n,
        rho=rho,
        logitscale=logitscale,
        condition_on_z=condition_on_z,
        epsilon=epsilon,
    )
    result["name"] = "backdoor_ot_z" if condition_on_z else "backdoor_ot"
    return result


if __name__ == "__main__":
    result = run(seed=0, n=200, rho=0.5, logitscale=1.0, condition_on_z=True, epsilon=0.0)
    print("\n===== Backdoor OT Smoke Test =====")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k:>20}: {v:.6f}")
        else:
            print(f"{k:>20}: {v}")
