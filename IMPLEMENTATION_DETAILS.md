# Experimental implementation details

This document is a code-first inventory of the implementation used by the paper examples, simulations, and 401(k) application. It is intended to serve as a compact experimental appendix. Each table links to the executable runner or the lowest-level implementation that fixes the stated value.

Values passed by a top-level runner take precedence over function defaults. A seed range such as 0–49 means the integer seeds produced by <code>range(50)</code>. Simulation code selects CUDA when available and otherwise uses CPU. Unless stated otherwise, no early stopping or gradient clipping is used.

## Experiment and entry-point index

| Paper item | Entry point(s) | Main sweep |
|---|---|---|
| Example 1: SCM noise misspecification | [SCM notebooks](examples/scm_example/) and [batch runner](examples/scm_example/run_scm_paper_examples.py#L26-L44) | Binary or mixed-tail noise; cocycle and Gaussian/Laplace-base causal flows |
| Example 2: Gaussian OT inconsistency | [OT notebook](examples/ot_example/OT_inconsistency.ipynb) | Direct versus composed Gaussian Brenier maps |
| Experiment 8.1 / Table 3 | [cocycles](simulations/linear_model/run_simlin_cocycles.py#L6-L37), [BGMs](simulations/linear_model/run_simlin_bgm.py#L6-L35) | 5 noises × 50 seeds; architecture cross-validation |
| Experiment 8.1 / Figure 10 | [fixed-linear runner](simulations/linear_model/run_simlin_linearfixed.py#L5-L22) | 5 noises × 50 seeds; common linear shift model |
| Experiment 8.2 / Design I | [cocycle](simulations/OT/run_simot_chain_cocycles.py#L5-L31), [OT and Seq-OT](simulations/OT/run_simot_chain_ot.py#L12-L53) | 5 correlations × 20 seeds; additive, multivariate Laplace noise |
| Experiment 8.2 / Design II | [cocycle](simulations/OT/run_simot_cocycles.py#L5-L31), [OT and Seq-OT](simulations/OT/run_simot_ot.py#L12-L54) | 5 correlations × 20 seeds; non-additive, independent Laplace noise |
| Experiment 8.3 | [cocycle](simulations/Csuite/run_simcsuite_cocycles.py#L6-L30), [CAREFL](simulations/Csuite/run_simcsuite_carefl.py#L6-L34), [CausalNF](simulations/Csuite/run_simcsuite_causalnf.py#L6-L34), [BGM](simulations/Csuite/run_simcsuite_bgm.py#L6-L34) | 8 SCMs × 10 seeds |
| 401(k) application | [notebook](applications/e401k-Cocycles-NF.ipynb) and [optimization config](applications/e401k_cocycle_config.py#L1-L13) | Full DoubleML 401(k) data; architecture cross-validation |

The <code>*_hpc.py</code> files implement cluster-parallel versions of the sweeps with the same primary settings. The Experiment 8.2 HPC launchers additionally include correct- and reversed-coordinate-order jobs ([Design I cocycle](simulations/OT/run_simot_chain_cocycles_hpc.py#L28-L63), [Design II cocycle](simulations/OT/run_simot_cocycles_hpc.py#L28-L63), [Seq-OT launchers](simulations/OT/run_simot_seqot_hpc.py#L28-L64)).

## Shared model and training components

### Candidate flow architectures

The architecture index saved in results has the following zero-based ordering. The same definitions are used by Experiments 8.1 and 8.3 and by the 401(k) application ([architecture factory](simulations/linear_model/architectures.py#L52-L133)). “Context” is the conditioning dimension supplied by the caller; adjacency masks are absent in the reported runs because <code>use_dag=False</code>.

| Index | Univariate transform(s) | Conditioner | Other settings | Selection LR |
|---:|---|---|---|---:|
| 0 | Shift only | Linear; no hidden layers | One autoregressive layer | 1e-2 |
| 1 | Shift only | MLP (32, 32) | One autoregressive layer | 1e-2 |
| 2 | Monotonic affine (location and scale) | MLP (32, 32) | One autoregressive layer | 1e-2 |
| 3 | Affine → rational-quadratic spline → affine | Separate MLP (32, 32) per layer | Spline shapes (8, 8, 9), i.e. 8 bins | 1e-3 |

Conditional cocycles use these transforms directly on the outcome. The BGM wraps each candidate in a coupling transform which leaves the input coordinate unchanged ([coupling factory](simulations/linear_model/architectures.py#L135-L192)). CAREFL and CausalNF apply autoregressive transforms to the complete observed vector with zero external context ([flow construction](simulations/Csuite/run_flows_mixed.py#L193-L213)).

Experiment 8.2 has a separate discrete-treatment architecture: treatment 0 is anchored to identity and treatments 1 and 2 have separate 2D monotonic-affine MAFs with (32, 32) hidden units. The current factory returns one candidate ([OT architecture](simulations/OT/architectures.py#L235-L263)).

### Objectives, kernels, validation, and optimization

| Component | Executable behavior | Source |
|---|---|---|
| CMMD kernel | Gaussian, k(a,b)=exp(-0.5‖a/ell-b/ell‖²). The length scale is sqrt(median(pairwise squared distances)/2), from at most 10,000 observations. | [kernel](causal_cocycle/kernels.py#L20-L24), [median heuristic](causal_cocycle/loss_factory.py#L116-L144) |
| CMMD variants | Experiments use the V- or U-statistic implementation named below. | [CMMD-V/U](causal_cocycle/loss_factory.py#L146-L213) |
| Cocycle selection | A 0.5 training fraction with method “CV” produces two complementary contiguous folds. The lowest mean validation-loss candidate is retrained on all data. | [selection](causal_cocycle/optimise_new.py#L206-L322) |
| Cocycle optimizer | Adam with PyTorch defaults and zero weight decay unless overridden. Each epoch performs max(floor(N/batch),1) updates, each from a fresh random subset. | [optimizer](causal_cocycle/optimise_new.py#L32-L186) |
| Likelihood-flow selection | Two-fold <code>KFold(shuffle=True)</code>; candidates minimize mean NLL with Adam, then the winner is retrained on all data. | [flow selector](causal_cocycle/causalflow_helper.py#L104-L178) |
| Evaluation truth | Experiments 8.1 and 8.3 use 100,000 paired observational/interventional SCM draws with shared exogenous noise. | [8.1](simulations/linear_model/run_cocycles.py#L17-L65), [8.3](simulations/Csuite/run_cocycles_mixed.py#L17-L57) |

Likelihood models use learnable factorized bases. Normal and Laplace initialize location 0 and scale <code>softplus(0)=log(2)</code>. Student-t initializes <code>raw_df=log(3)</code>, giving implemented initial degrees of freedom <code>softplus(log(3))+0.01=log(4)+0.01</code>; its location and scale initialize as above ([base implementations](causal_cocycle/causalflow_helper.py#L18-L101)).

## Example 1: SCM noise misspecification

All cases use Y=(X+1)U, hence Y(0)=U, Y(1)=2U, and the true individual effect is U.

| Noise | Distribution of U | Training construction | Source |
|---|---|---|---|
| Binary | Bernoulli(0.5) | 1,000 observations per treatment; each arm has 500 zeros and 500 ones. | [cocycle](examples/scm_example/cocycles_binary_example.ipynb), [Gaussian flow](examples/scm_example/gaussian_flow_binary_example.ipynb) |
| Mixed tails | With probability 1/2, abs(N(0,1)); otherwise -abs(N(0,tau)), tau distributed BetaPrime(0.1,0.1) | 1,000 observations per treatment. The cocycle draws 2,000 noises for pooled treatment data; flows construct Y(1)=2Y(0) from 1,000 control draws. | [cocycle](examples/scm_example/cocycles_mixedtails_example.ipynb), [Gaussian flow](examples/scm_example/gaussian_flow_mixedtails_example.ipynb) |

| Method | Architecture / objective | Optimization | Evaluation |
|---|---|---|---|
| Cocycle | Treatment-0 identity; treatment-1 unconditional 1D monotonic-affine transform with no hidden layers; CMMD-V | Adam, LR 1e-2, 1,000 epochs, batch 128, no validation | 100,000 transported controls; training seed 0, truth seed 2026 |
| Gaussian-base flow | Separate unconditional affine–RQS–affine flow per arm; fixed N(0,1) base; NLL | Adam, LR 1e-2, 1,000 epochs, batch 128 | Shared base draw of 100,000; effect seed 17, truth seed 2026 |
| Laplace-base flow | Same with fixed Laplace(0,1) bases; NLL | Adam, 1,000 epochs, batch 128; LR 1e-3 for binary and 1e-2 for mixed tails | Shared base draw of 100,000 |

The flow is defined in [the example architecture](examples/scm_example/architectures.py#L9-L56). The batch runner executes the two cocycle and two Gaussian-base notebooks used for the combined figure; the Laplace notebooks are separately executable ([notebook list](examples/scm_example/run_scm_paper_examples.py#L26-L32)).

## Example 2: Gaussian OT inconsistency

| Setting | Value |
|---|---|
| Distributions | P0=N((0,0),I); P1=N((1,1),[[1,-0.9],[-0.9,1]]); P2=N((-1,2),diag(0.5,5)) |
| Sample | 200 draws from P0; NumPy seed 0 |
| Map | Closed-form affine Gaussian Brenier map T(x)=m′+A(x-m), with A=S^(-1/2)(S^(1/2)S′S^(1/2))^(1/2)S^(-1/2) |
| Comparison | Direct T02 versus T12∘T01; effect is transported outcome minus the original P0 draw |

All settings are in [the Example 2 notebook](examples/ot_example/OT_inconsistency.ipynb).

## Experiment 8.1: noise ablation in a linear model

### Architecture-selection experiment (Table 3)

| Setting | Value | Source |
|---|---|---|
| SCM | X1=U1; X2=10X1-U2; X3=0.25X2+2U3; X4=X3+U4; X5=-X4+U5 | [SCM](simulations/linear_model/csuite.py#L246-L277) |
| Replicates | N=1,000; seeds 0–49 | [launcher](simulations/linear_model/run_simlin_cocycles.py#L7-L29) |
| Noise | U1~N(0,1). Independently for U2:5: standard Normal, Rademacher from sign(Uniform(-1,1)), standard Cauchy, Gamma(1,1), or inverse-Gamma(1,1). Correlation is zero. | [construction](simulations/linear_model/run_cocycles.py#L189-L212) |
| Modeling | Input X1; outcome (X2,…,X5); no DAG mask; intervention do(X1=0) | [setup](simulations/linear_model/run_cocycles.py#L183-L225) |

| Estimator | Candidates / base | Objective and selection | Training |
|---|---|---|---|
| Cocycle CMMD-V | Indices 0–3 | CMMD-V; two-fold selection and full-data retraining | Adam; 1,000 epochs; batch 128; LR 1e-2 for 0–2 and 1e-3 for 3 |
| Cocycle CMMD-U | Indices 0–3 | CMMD-U; otherwise identical | Same |
| BGM-N | Coupled indices 0–3; learnable Normal base | NLL; shuffled two-fold selection and retraining | Adam; 1,000 epochs; batch 128; CV rates 1e-2 for 0–2 and 1e-3 for 3 |
| BGM-L | Same; learnable Laplace base | Same | Same |
| BGM-T | Same; learnable Student-t base | Same | Same |

See [cocycle implementation](simulations/linear_model/run_cocycles.py#L153-L261) and [BGM implementation](simulations/linear_model/run_bgm.py#L172-L278).

### Fixed-linear experiment (Figure 10)

| Component | Implemented value |
|---|---|
| DGP | X~N(1,1), Y=X+U, true slope 1; N=1,000; seeds 0–49 |
| Noise | Standard Normal; Rademacher; standard Cauchy; Gamma(1,1)-1; inverse-Gamma(1,1), uncentered |
| Architecture | Linear conditioner without bias followed by a shift |
| Estimators | ML with learnable Normal or Laplace base; URR with either base; CMMD-V; CMMD-U; true coefficient |
| Optimization | Full sample; Adam; LR 1e-2; 1,000 epochs; batch 128; no scheduler |
| Output | Learned linear coefficient |

See [the fixed-linear implementation](simulations/linear_model/run_linear_model.py#L14-L143).

### Evaluation metrics

Metrics are coordinatewise for X2:5. <code>KS_CF</code> and <code>W1_CF</code> compare estimated and true paired-change distributions; <code>KS_int</code> and <code>W1_int</code> compare interventional marginals; <code>RMSE_CF</code> compares paired counterfactual outcomes. Distribution metrics use 100,000 truth draws and paired RMSE uses the 1,000 training units and their SCM counterfactuals ([evaluation](simulations/linear_model/run_cocycles.py#L67-L149), [definitions](causal_cocycle/helper_functions.py#L71-L112)).

## Experiment 8.2: confounding and path-consistency ablation

For domain x in {0,1,2}, the 2D outcome is Y(x)=m_x+xi_x L_xᵀ, with means (0,0), (1,1), (2,2) and L_x the Cholesky factor of S_x. Truth for domain-0 units reuses standardized noise under counterfactual domains ([DGP](simulations/OT/dgp.py#L53-L130)).

| Setting | Design I: additive, multivariate Laplace | Design II: non-additive, independent Laplace |
|---|---|---|
| S0 | I | I |
| S1 | I | [[1,-rho],[-rho,1]] |
| S2 | I | diag(1+rho,1/(1+rho)) |
| Raw noise | sqrt(W)Z, W~Exp(1), Z~N(0,[[1,rho],[rho,1]]), then xi2←xi1+xi2 | Independent Laplace(0,1) coordinates |
| Sweep | 500 observations per domain; rho in {0.1,0.3,0.5,0.7,0.9}; seeds 0–19 | Same |

The multivariate-Laplace sampler is in [helpers](simulations/OT/helpers.py#L4-L42).

| Method | Implementation | Hyperparameters |
|---|---|---|
| Cocycle | Identity anchor at treatment 0; monotonic-affine MAFs for 1 and 2; CMMD-U | Random permutation, fixed 50:50 validation, full-data retraining; Adam, LR 1e-2, 1,000 epochs, batch 128 ([runner](simulations/OT/run_cocycles.py#L31-L89)) |
| OT | Exact empirical OT with uniform weights, squared-Euclidean cost, and <code>ot.emd</code>; barycentric pairwise projections | No learned hyperparameters ([implementation](simulations/OT/run_ot.py#L13-L57)) |
| Seq-OT | Empirical 1D monotone map for coordinate 1; conditional weighted quantile map for coordinate 2 with a median-bandwidth Gaussian kernel | Smoothing epsilon 0. Design II uses arm-specific source/target samples. Design I column-stacks the arms, but the conditional arrays selected as <code>Y[:,0]</code> and <code>Y[:,1]</code> are the two control-arm columns; treatment-specific first-coordinate KR maps determine the query locations ([Design I](simulations/OT/run_seqot_chain.py#L11-L102), [Design II](simulations/OT/run_seqot.py#L11-L97)) |
| Reversed order | Flip the two coordinates before cocycle or Seq-OT fitting | HPC launchers cited above |

Fields named <code>RMSE*</code> are mean per-unit Euclidean effect-error norms, not square roots of scalar MSEs. <code>ATE*</code> is root mean squared coordinatewise effect bias. <code>RMSEinconsistency</code> is mean Euclidean distance between direct and composed maps ([cocycle](simulations/OT/run_cocycles.py#L91-L117), [OT](simulations/OT/run_ot.py#L64-L97), [Seq-OT](simulations/OT/run_seqot.py#L99-L155)).

## Experiment 8.3: SCM benchmark

Every SCM uses N=2,000, seeds 0–9, input X1, outcomes X2:d, no DAG mask, and do(X1=0). Evaluation uses 100,000 paired truth draws ([launcher](simulations/Csuite/run_simcsuite_cocycles.py#L6-L25), [evaluation](simulations/Csuite/run_cocycles_mixed.py#L17-L137)).

### Exogenous noise

Noises are mutually independent ([generator](simulations/Csuite/csuite_mixed.py#L27-L50)).

| Dimension | Noise sequence (U1,…,Ud) |
|---:|---|
| 2 | N(0,1); inverse-Gamma(1,1) |
| 3 | N(0,1); Rademacher; inverse-Gamma(1,1) |
| 4 | N(0,1); Gamma(1,1); Rademacher; inverse-Gamma(1,1) |
| 5 | N(0,1); equal mixture N(-sqrt(3)/2,0.5²)/N(sqrt(3)/2,0.5²); Gamma(1,1); Rademacher; inverse-Gamma(1,1) |

### Structural equations

| SCM | Equations | Source |
|---|---|---|
| 2-variable linear | X1=U1; X2=X1+U2 | [code](simulations/Csuite/csuite_mixed.py#L82-L107) |
| 2-variable nonlinear | X1=U1; X2=sin(X1)+U2 | [code](simulations/Csuite/csuite_mixed.py#L109-L133) |
| Triangle linear | X1=U1+1; X2=10X1-U2; X3=0.5X2+X1+U3 | [code](simulations/Csuite/csuite_mixed.py#L136-L161) |
| Triangle nonlinear | X1=U1+1; X2=2X1²+U2; X3=20(1+exp(-X2²+X1))+U3 | [code](simulations/Csuite/csuite_mixed.py#L163-L189) |
| Fork linear | X1=U1; X2=2-U2; X3=0.25X2-1.5X1+0.5U3; X4=X3+0.25U4 | [code](simulations/Csuite/csuite_mixed.py#L191-L218) |
| Fork nonlinear | X1=U1; X2=U2; X3=4/(1+exp(-X1-X2))-X2²+0.5U3; X4=20/(1+exp(0.5X3²-X3))+U4 | [code](simulations/Csuite/csuite_mixed.py#L220-L247) |
| 5-chain linear | X1=U1; X2=10X1-U2; X3=0.25X2+2U3; X4=X3+U4; X5=-X4+U5 | [code](simulations/Csuite/csuite_mixed.py#L250-L281) |
| 5-chain nonlinear | X1=tanh(U1); X2=X1²+U2; X3=sin(X2)+U3; X4=X3X2+U4; X5=exp(-X4)+U5 | [code](simulations/Csuite/csuite_mixed.py#L283-L313) |

### Estimators

| Label | Fitted object | Candidates | Base/objective | Optimization |
|---|---|---|---|---|
| Cocycle | Conditional map X1→X2:d | 0–3 | CMMD-V | Two-fold cocycle selection; Adam, 1,000 epochs, batch 128, rates 1e-2/1e-3; retrain ([implementation](simulations/Csuite/run_cocycles_mixed.py#L140-L209)) |
| CAREFL | Autoregressive flow on full d-vector | 0–2; RQS removed by <code>affine=True</code> | Learnable Normal, Laplace, Student-t; NLL | Shuffled two-fold selection; Adam, 1,000 epochs, batch 128, LR 1e-2 ([wrapper](simulations/Csuite/run_simcsuite_carefl.py#L6-L34), [implementation](simulations/Csuite/run_flows_mixed.py#L158-L236)) |
| CausalNF | Autoregressive flow on full d-vector | 0–3 | Same bases; NLL | Same; candidate 3 CV rate 1e-3 ([wrapper](simulations/Csuite/run_simcsuite_causalnf.py#L6-L34)) |
| BGM | Coupling flow leaving X1 unchanged and conditionally transforming X2:d | 0–3 | Same bases; NLL | Same likelihood selection ([implementation](simulations/Csuite/run_bgm_mixed.py#L159-L238)) |

All methods report coordinatewise KS/W1 paired-change and interventional metrics plus paired counterfactual RMSE, as defined under Experiment 8.1.

## 401(k) application

| Component | Implemented value | Source |
|---|---|---|
| Data | <code>fetch_401K(return_type='DataFrame')</code>; saved run has 9,915 records and 3,682 eligible | [notebook](applications/e401k-Cocycles-NF.ipynb) |
| Treatment/outcome | <code>e401</code> / <code>net_tfa</code> | [notebook](applications/e401k-Cocycles-NF.ipynb) |
| Model inputs | age, inc, educ, fsize, marr, twoearn, db, pira, hown, plus e401 moved first | [notebook](applications/e401k-Cocycles-NF.ipynb) |
| Scaling | Divide non-binary inputs and the outcome by their sample SD; binary inputs unchanged; no centering | [notebook](applications/e401k-Cocycles-NF.ipynb) |
| Data use | Random permutation, then 100% used for selection and fitting; no held-out test set | [notebook](applications/e401k-Cocycles-NF.ipynb) |
| Model | Scalar-outcome conditional cocycle with all 10 inputs as context; candidates 0–3; CMMD-V | [notebook](applications/e401k-Cocycles-NF.ipynb), [architecture](applications/architectures.py#L52-L133) |
| Training | Two folds and retraining; Adam; 1,000 epochs; batch 128; LR 1e-2 for 0–2 and 1e-3 for 3; weight decay 1e-3; StepLR every epoch, multiplier 0.9 | [config](applications/e401k_cocycle_config.py#L1-L13) |
| Effects | Set treatment to 0 and 1 for every unit; evaluate relative to observed treatment/outcome; rescale by outcome SD. ATE uses all units and ATT observed treated units. | [notebook](applications/e401k-Cocycles-NF.ipynb) |
| Conditional summaries | Nadaraya–Watson regression on rank income or rank predicted Y(0); Gaussian kernel initialized at length scale 1, regularization 0; 5 folds, 1,000 iterations, LR 0.1, subsample 256; 500 prediction points on [0.1,0.9] | [notebook](applications/e401k-Cocycles-NF.ipynb) |

## Software environment

The supplied environment fixes Python 3.12.2, NumPy 1.26, PyTorch 2.1.2, Matplotlib 3.8.3, pandas 2.2.0, DoubleML 0.7.1, causalflows 0.1.0, Zuko 1.4.0, and seaborn 0.13.2 ([environment](environment.yml)).

## Current-code verification notes

1. Cocycle runner arguments named <code>k_folds</code> are not passed to <code>validate</code>; the 0.5 split fixes the actual count at two ([8.1](simulations/linear_model/run_cocycles.py#L153-L164), [8.3](simulations/Csuite/run_cocycles_mixed.py#L140-L149)).
2. The Experiment 8.2 builder comment refers to four architectures, but the current factory returns one anchored affine-MAF candidate ([builder](simulations/OT/run_cocycles.py#L11-L20), [factory](simulations/OT/architectures.py#L235-L263)).
3. Likelihood-flow CV uses the candidate-specific RQS rate 1e-3, but final retraining uses the base LR 1e-2 for whichever candidate wins ([selector](causal_cocycle/causalflow_helper.py#L137-L173)).
4. Local Experiment 8.2 launchers use correct coordinate order; reversed-order ablations are in the HPC launchers.
5. The 401(k) notebook does not seed its random permutation. Its later quantile cell refers to <code>NWConditioner</code> and <code>KREpsLayer</code> without importing them in the notebook, so those plotting cells require the classes in the active kernel namespace.
6. The 401(k) quantile computation requests (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975), while the plot labels these curves as (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95). Cocycle fitting, ATE, and ATT precede and do not depend on this plotting-label mismatch.
7. In Design I Seq-OT, <code>torch.column_stack((Y0,Y1,Y2))</code> creates six columns rather than pooling rows. The subsequent conditional-map code uses only columns 0 and 1, so the implementation behavior is the control-arm construction stated in the method table.
