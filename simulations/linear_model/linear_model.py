from torch.distributions import Normal, Laplace, Uniform, Gamma, Cauchy
import torch
from causal_cocycle.model_new import CocycleModel, FlowModel
from causal_cocycle.optimise_new import optimise
from causal_cocycle.loss import CocycleLoss
from causal_cocycle.loss_factory import CocycleLossFactory
from causal_cocycle.conditioners_new import LinConditioner, CompositeConditioner
from causal_cocycle.transformers_new import Transformer, ShiftLayer
from causal_cocycle.kernels import gaussian_kernel as GaussianKernel
from causal_cocycle.loss import FlowLoss
from causal_cocycle.causalflow_helper import LearnableNormal, LearnableLaplace, LearnableStudentT


def run_experiment(seed, N, noise_type="rademacher"):
    """
    Function to run regression estimation experiment with updated cocycle model.
    """
    N, D, P = N, 1, 1
    sig_noise_ratio = 1

    train_val_split = 1
    ntrain = int(train_val_split * N)
    learn_rate = [1e-2]
    epochs = 1000
    scheduler = False
    val_tol = 1e-3
    batch_size = 128
    val_loss = False
    bias = False

    names = [
      "ML-N", "ML-L", "ML-T",
      "URR-N", "URR-L", "URR-T",
      "CMMD-V", "CMMD-U",
      "True"
    ]    
    Coeffs = torch.zeros((1, len(names), P))

    # Data
    torch.manual_seed(seed)
    X = Normal(1, 1).sample((N, D)) / D**0.5
    B = torch.ones((D, 1)) * (torch.linspace(0, D - 1, D) < P)[:, None]
    F = X @ B

    if noise_type == "normal":
        U = Normal(0, 1).sample((N, 1))
    elif noise_type == "rademacher":
        U = torch.sign(Uniform(-1, 1).sample((N, 1)))
    elif noise_type == "cauchy":
        U = Cauchy(0, 1).sample((N, 1))
    elif noise_type == "gamma":
        U = Gamma(1.0, 1.0).sample((N, 1))-1
    elif noise_type == "inversegamma":
        U = 1.0 / Gamma(1.0, 1.0).sample((N, 1))
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    U = U / sig_noise_ratio**0.5
    Y = F + U

    def train_model(loss_type,
                    X, Y,
                    model_class=CocycleModel,
                    base="Normal",
                    use_flow_loss=False):
        """
        loss_type: str, one of {"HSIC","URR","CMMD_V","CMMD_U"}, or "L1" if
                   use_flow_loss=True
        use_flow_loss: bool, if True, train a FlowModel with FlowLoss(log_det=False)
        """
        # build conditioner & transformer
        conditioner = CompositeConditioner([LinConditioner(D, 1, bias=bias)])
        transformer = Transformer([ShiftLayer()])
    
        # instantiate the model
        if model_class is FlowModel:
            # choose base distribution
            if base == "Normal":
                base_dist = LearnableNormal(dim = 1)
            elif base == "Laplace":
                base_dist = LearnableLaplace(dim = 1)
            elif base == "StudentT":
                base_dist = LearnableStudentT(dim = 1)        
            else:
                raise ValueError(f"Unsupported base: {base}")
    
            model = FlowModel(conditioner, transformer, base_dist)
        else:
            model = CocycleModel(conditioner, transformer)
    
        # pick loss function
        if model_class is FlowModel and use_flow_loss:
            # e.g. for L1‐flow
            loss_fn = FlowLoss(log_det=False)
        else:
            # all other losses via kernel/cocycle API
            kernel = [GaussianKernel(), GaussianKernel()]
            loss_factory = CocycleLossFactory(kernel)
            loss_fn = loss_factory.build_loss(loss_type, X, Y, subsamples=10**4)
    
        # train
        optimise(
            model,
            loss_fn,
            X[:ntrain],
            Y[:ntrain],
            batch_size=batch_size,
            learn_rate=learn_rate[0],
            print_=True,
            plot=False,
            epochs=epochs,
            scheduler=scheduler
        )
        return model


    # ML
    ML_normal = train_model("ML", X, Y, model_class=FlowModel, base="Normal",use_flow_loss=True)
    ML_laplace = train_model("ML", X, Y, model_class=FlowModel, base="Laplace",use_flow_loss=True)
    ML_studentT = train_model("ML", X, Y, model_class=FlowModel, base="StudentT",use_flow_loss=True)


    # URR
    URR_normal = train_model("URR", X, Y, model_class=FlowModel, base="Normal",use_flow_loss=False)
    URR_laplace = train_model("URR", X, Y, model_class=FlowModel, base="Laplace",use_flow_loss=False)
    URR_studentT = train_model("URR", X, Y, model_class=FlowModel, base="StudentT",use_flow_loss=False)

    # Cocycles
    CMMDV = train_model("CMMD_V", X, Y)
    CMMDU = train_model("CMMD_U", X, Y)    
  
    # Coefficient extraction
    def extract_weight(model):
        return model.conditioner.layers[0].linear.weight.data

    Coeffs[:, 0] = extract_weight(ML_normal)
    Coeffs[:, 1] = extract_weight(ML_laplace)
    Coeffs[:, 2] = extract_weight(ML_studentT)
    Coeffs[:, 3] = extract_weight(URR_normal)
    Coeffs[:, 4] = extract_weight(URR_laplace)
    Coeffs[:, 5] = extract_weight(URR_studentT)
    Coeffs[:, 6] = extract_weight(CMMDV)
    Coeffs[:, 7] = extract_weight(CMMDU)
    Coeffs[:, 8] = B.T
    
    return {
        "seed": seed,
        "names": names,
        "Coeffs": Coeffs,
        "dist": noise_type,
        "nsamples": N
    }


if __name__ == "__main__":
    results = run_experiment(seed=0, N=1000, noise_type="rademacher")
    for name, coeff in zip(results['names'], results['Coeffs'][0]):
        print(f"{name}: {coeff.item():.4f}")
