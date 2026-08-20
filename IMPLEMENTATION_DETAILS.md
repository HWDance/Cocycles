# Experimental implementation details

This document is a code-first inventory of the implementation used by the paper examples, simulations, and 401(k) application. It is intended to serve as a compact experimental appendix. Each table links to the executable runner or the lowest-level implementation that fixes the stated value.

Values passed by a top-level runner take precedence over function defaults. A seed range such as 0–49 means the integer seeds produced by <code>range(50)</code>. Simulation code selects CUDA when available and otherwise uses CPU. Unless stated otherwise, no early stopping or gradient clipping is used.

## Experiment and entry-point index

| Paper item | Entry point(s) | Main sweep |
|---|---|---|
| Experiment 8.1 / binary and mixed-tail noise | [SCM notebooks](examples/scm_example/) and [batch runner](examples/scm_example/run_scm_paper_examples.py#L26-L44) | Binary or mixed-tail noise; cocycle and Gaussian/Laplace-base causal flows |
| Example 2: Gaussian OT inconsistency | [OT notebook](examples/ot_example/OT_inconsistency.ipynb) | Direct versus composed Gaussian Brenier maps |
| Experiment 8.1 / Table 3 | [cocycles](simulations/linear_model/run_simlin_cocycles.py#L6-L37), [BGMs](simulations/linear_model/run_simlin_bgm.py#L6-L35) | 5 noises × 50 seeds; architecture cross-validation |
| Experiment 8.1 / Figure 10 | [fixed-linear runner](simulations/linear_model/run_simlin_linearfixed.py#L5-L22) | 5 noises × 50 seeds; common linear shift model |
| Experiment 8.2 / Design I | [cocycle](simulations/OT/run_simot_chain_cocycles.py#L5-L31), [OT and Seq-OT](simulations/OT/run_simot_chain_ot.py#L12-L53) | 5 correlations × 20 seeds; additive, multivariate Laplace noise |
| Experiment 8.2 / Design II | [cocycle](simulations/OT/run_simot_cocycles.py#L5-L31), [OT and Seq-OT](simulations/OT/run_simot_ot.py#L12-L54) | 5 correlations × 20 seeds; non-additive, independent Laplace noise |
| Experiment 8.3 | [cocycle](simulations/Csuite/run_simcsuite_cocycles.py#L6-L30), [CAREFL](simulations/Csuite/run_simcsuite_carefl.py#L6-L34), [CausalNF](simulations/Csuite/run_simcsuite_causalnf.py#L6-L34), [BGM](simulations/Csuite/run_simcsuite_bgm.py#L6-L34) | 8 SCMs × 10 seeds |
| 401(k) application | [notebook](applications/e401k-Cocycles-NF.ipynb) and [optimization config](applications/e401k_cocycle_config.py#L1-L13) | Full DoubleML 401(k) data; architecture cross-validation |

The <code>*_hpc.py</code> files implement cluster-parallel versions of the sweeps with the same primary settings. The Experiment 8.2 HPC launchers additionally include correct- and reversed-coordinate-order jobs ([Design I cocycle](simulations/OT/run_simot_chain_cocycles_hpc.py#L28-L63), [Design II cocycle](simulations/OT/run_simot_cocycles_hpc.py#L28-L63), [Seq-OT launchers](simulations/OT/run_simot_seqot_hpc.py#L28-L64)).

## Architecture dictionary used in the experiments

The code stores the selected architecture as an integer from 0 to 3. To make those saved indices interpretable, this document calls the four architectures A0–A3. The definitions below are the complete search space used by the linear-chain simulation, the SCM benchmark, and the 401(k) application ([architecture factory](simulations/linear_model/architectures.py#L52-L133)).

| Code index / label | Exact architecture | Network depth and width | Flow depth / spline settings | CV learning rate |
|---:|---|---|---|---:|
| 0 / A0 | Linear shift MAF: each output is transformed by a learned shift; no learned scale | No hidden layer (<code>hidden_features=()</code>) | One masked-autoregressive transform | 1e-2 |
| 1 / A1 | Neural shift MAF: shift-only transformation | Two hidden layers, width 32 each | One masked-autoregressive transform | 1e-2 |
| 2 / A2 | Neural affine MAF: learned location and positive scale | Two hidden layers, width 32 each | One masked-autoregressive transform | 1e-2 |
| 3 / A3 | Neural spline MAF: affine → rational-quadratic spline → affine | Each of the three transforms has two hidden layers, width 32 each | Three masked-autoregressive transforms; RQS has 8 bins (parameter shapes 8, 8, and 9) | 1e-3 |

The learning-rate column describes the full A0–A3 search. CAREFL removes A3, after which the generic likelihood selector assigns its reduced search's last candidate, A2, the smaller 1e-3 cross-validation rate. All likelihood winners are retrained at the base rate 1e-2.

“MAF” here refers to Zuko's <code>MaskedAutoregressiveTransform</code>. Conditional cocycles apply A0–A3 to the outcome with the treatment/input as context. BGM wraps the same outcome MAF in a coupling transform that leaves the input coordinate unchanged ([coupling factory](simulations/linear_model/architectures.py#L135-L192)). CAREFL and CausalNF apply the MAF to the complete observed vector with zero external context ([flow construction](simulations/Csuite/run_flows_mixed.py#L193-L213)). The reported runs set <code>use_dag=False</code>, so none of these transforms receives an adjacency mask.

| Use | MAF features | Context | Interpretation |
|---|---:|---:|---|
| Experiment 8.1 linear-chain cocycle | 4 | 1 | Transform (X2,…,X5) conditional on X1 |
| Experiment 8.1 linear-chain BGM | 4 | 1 | Coupling flow leaves X1 fixed and transforms (X2,…,X5) |
| Experiment 8.3 cocycle | d-1 | 1 | Transform X2:d conditional on X1 |
| Experiment 8.3 CAREFL/CausalNF | d | 0 | Autoregressive flow over the complete d-dimensional vector |
| Experiment 8.3 BGM | d-1 | 1 | Coupling flow leaves X1 fixed and transforms X2:d |
| 401(k) cocycle | 1 | 10 | Transform scalar net financial assets conditional on treatment and nine covariates |

Experiment 8.2 has a separate discrete-treatment architecture: treatment 0 is anchored to identity and treatments 1 and 2 have separate 2D monotonic-affine MAFs with (32, 32) hidden units. The current factory returns one candidate ([OT architecture](simulations/OT/architectures.py#L235-L263)).

## Shared objectives and training conventions

| Component | Executable behavior | Source |
|---|---|---|
| CMMD kernel | Gaussian, k(a,b)=exp(-0.5‖a/ell-b/ell‖²). The length scale is sqrt(median(pairwise squared distances)/2), from at most 10,000 observations. | [kernel](causal_cocycle/kernels.py#L20-L24), [median heuristic](causal_cocycle/loss_factory.py#L116-L144) |
| CMMD variants | Experiments use the V- or U-statistic implementation named below. | [CMMD-V/U](causal_cocycle/loss_factory.py#L146-L213) |
| Cocycle selection | A 0.5 training fraction with method “CV” produces two complementary contiguous folds. The lowest mean validation-loss candidate is retrained on all data. | [selection](causal_cocycle/optimise_new.py#L206-L322) |
| Cocycle optimizer | Adam with PyTorch defaults and zero weight decay unless overridden. Each epoch performs max(floor(N/batch),1) updates, each from a fresh random subset. | [optimizer](causal_cocycle/optimise_new.py#L32-L186) |
| Likelihood-flow selection | Two-fold <code>KFold(shuffle=True)</code>; candidates minimize mean NLL with Adam, then the winner is retrained on all data. | [flow selector](causal_cocycle/causalflow_helper.py#L104-L178) |
| Evaluation truth | Experiments 8.1 and 8.3 use 100,000 paired observational/interventional SCM draws with shared exogenous noise. | [8.1](simulations/linear_model/run_cocycles.py#L17-L65), [8.3](simulations/Csuite/run_cocycles_mixed.py#L17-L57) |

Likelihood models use learnable factorized bases. Normal and Laplace initialize location 0 and scale <code>softplus(0)=log(2)</code>. Student-t initializes <code>raw_df=log(3)</code>, giving implemented initial degrees of freedom <code>softplus(log(3))+0.01=log(4)+0.01</code>; its location and scale initialize as above ([base implementations](causal_cocycle/causalflow_helper.py#L18-L101)).

## Example 2: Gaussian OT inconsistency

| Setting | Value |
|---|---|
| Distributions | P0=N((0,0),I); P1=N((1,1),[[1,-0.9],[-0.9,1]]); P2=N((-1,2),diag(0.5,5)) |
| Sample | 200 draws from P0; NumPy seed 0 |
| Map | Closed-form affine Gaussian Brenier map T(x)=m′+A(x-m), with A=S^(-1/2)(S^(1/2)S′S^(1/2))^(1/2)S^(-1/2) |
| Comparison | Direct T02 versus T12∘T01; effect is transported outcome minus the original P0 draw |

All settings are in [the Example 2 notebook](examples/ot_example/OT_inconsistency.ipynb).

## Experiment 8.1: noise misspecification and linear-model ablation

### Binary and mixed-tail structural model

This experiment was called “Example 1” in the repository README. In the current paper organization it forms the first part of Experiment 8.1. All cases use Y=(X+1)U, hence Y(0)=U, Y(1)=2U, and the true individual effect is U.

| DGP | Exogenous noise U | Training sample |
|---|---|---|
| Binary | Bernoulli(0.5) | 1,000 observations per treatment; each arm contains 500 zeros and 500 ones. |
| Mixed tails | Draw B~Bernoulli(0.5). If B=1, U=abs(Z) with Z~N(0,1). If B=0, draw tau~BetaPrime(0.1,0.1), Z~N(0,tau), and set U=-abs(Z). | 1,000 observations per treatment. The cocycle notebook draws 2,000 independent U values for pooled treatment data; the flow notebooks draw 1,000 controls and set Y(1)=2Y(0). |

| Estimator | Architecture | Objective | Optimization | Effect evaluation |
|---|---|---|---|---|
| Cocycle | Discrete selector with identity map at X=0 and one unconditional scalar monotonic-affine transform at X=1; no hidden layers | CMMD-V with median-heuristic Gaussian output kernel | Adam; LR 1e-2; 1,000 epochs; batch 128; all 2,000 observations; no validation split | Transport 100,000 control outcomes from X=0 to X=1; training seed 0, truth seed 2026 |
| Gaussian-base causal flow | Separate scalar flow for each treatment arm. Each flow has three transforms: affine → 8-bin RQS → affine; all are unconditional (<code>hidden_features=()</code>); fixed N(0,1) base | Negative log likelihood | Adam; LR 1e-2; 1,000 epochs; batch 128; shuffled DataLoader | Apply both fitted inverse flows to the same 100,000 base draws; effect seed 17, truth seed 2026 |
| Laplace-base causal flow | Same separate affine → 8-bin RQS → affine flows with fixed Laplace(0,1) bases | Negative log likelihood | Adam; 1,000 epochs; batch 128; LR 1e-3 for binary noise and 1e-2 for mixed tails | Apply both fitted inverse flows to the same 100,000 base draws |

The data and fitting code are in the [binary cocycle](examples/scm_example/cocycles_binary_example.ipynb), [mixed-tail cocycle](examples/scm_example/cocycles_mixedtails_example.ipynb), [Gaussian-base](examples/scm_example/gaussian_flow_binary_example.ipynb), and [Laplace-base](examples/scm_example/laplace_flow_binary_example.ipynb) notebooks. The 1D flow is defined in [the example architecture](examples/scm_example/architectures.py#L9-L56). The batch runner executes the two cocycle and two Gaussian-base notebooks used for the combined figure; the Laplace notebooks are separately executable ([notebook list](examples/scm_example/run_scm_paper_examples.py#L26-L32)).

### Five-node linear chain with architecture selection (Table 3)

| Setting | Value | Source |
|---|---|---|
| SCM | X1=U1; X2=10X1-U2; X3=0.25X2+2U3; X4=X3+U4; X5=-X4+U5 | [SCM](simulations/linear_model/csuite.py#L246-L277) |
| Replicates | N=1,000; seeds 0–49 | [launcher](simulations/linear_model/run_simlin_cocycles.py#L7-L29) |
| Noise | U1~N(0,1). Independently for U2:5: standard Normal, Rademacher from sign(Uniform(-1,1)), standard Cauchy, Gamma(1,1), or inverse-Gamma(1,1). Correlation is zero. | [construction](simulations/linear_model/run_cocycles.py#L189-L212) |
| Modeling | Input X1; outcome (X2,…,X5); no DAG mask; intervention do(X1=0) | [setup](simulations/linear_model/run_cocycles.py#L183-L225) |

| Estimator | Exact architecture search | Base and objective | Model selection | Optimization budget |
|---|---|---|---|---|
| Cocycle CMMD-V | Conditional 4D MAF for (X2,…,X5) given scalar X1. Search A0 linear shift, A1 two-layer 32-wide neural shift, A2 two-layer 32-wide neural affine, and A3 three-transform affine–8-bin-RQS–affine with a two-layer 32-wide network in every transform. | No latent base distribution; CMMD-V with median-heuristic Gaussian output kernel | Two complementary folds; select lowest mean validation CMMD-V; retrain on all 1,000 observations | Adam; 1,000 epochs in every fold and in retraining; batch 128; LR 1e-2 for A0–A2 and 1e-3 for A3; weight decay 0; no scheduler |
| Cocycle CMMD-U | Same conditional A0–A3 search | No latent base distribution; CMMD-U | A separate two-fold selection and retraining run using CMMD-U | Same |
| BGM-N | Coupling flow on the 5D vector: X1 is unchanged and the conditional 4D outcome transform searches the same A0–A3 MAFs | Learnable factorized Normal base on all five coordinates; negative log likelihood | Shuffled two-fold NLL selection; retrain on all 1,000 observations | Adam; 1,000 epochs in every fold and in retraining; batch 128; CV LR 1e-2 for A0–A2 and 1e-3 for A3; weight decay 0; no scheduler |
| BGM-L | Same coupling-flow A0–A3 search | Learnable factorized Laplace base; negative log likelihood | Separate architecture selection for this base | Same |
| BGM-T | Same coupling-flow A0–A3 search | Learnable factorized Student-t base; negative log likelihood | Separate architecture selection for this base | Same |

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

Design I generates each arm's exogenous vector independently as follows. First draw W~Exponential(rate=1) and Z=(Z1,Z2)~N(0,R_rho), where R_rho has ones on the diagonal and rho off the diagonal. Form V=sqrt(W)Z, then set xi=(V1,V1+V2). Thus rho controls dependence in the Gaussian component, and the final assignment adds the first component into the second. Since S0=S1=S2=I, treatment changes only the mean in this design. Design II instead draws the two components of xi independently from Laplace(0,1), while treatment changes the structural Cholesky factor.

| Setting | Design I: additive, multivariate Laplace | Design II: non-additive, independent Laplace |
|---|---|---|
| S0 | I | I |
| S1 | I | [[1,-rho],[-rho,1]] |
| S2 | I | diag(1+rho,1/(1+rho)) |
| Exogenous draw xi | Draw W~Exponential(rate=1) and Z~N(0,R_rho); set V=sqrt(W)Z and xi=(V1,V1+V2). Each treatment arm uses an independent draw with RNG seeds s, s+1, and s+2. | Draw xi1 and xi2 independently from Laplace(0,1), again separately by arm. |
| Sweep | 500 observations per domain; rho in {0.1,0.3,0.5,0.7,0.9}; seeds 0–19 | Same |

The multivariate-Laplace sampler is in [helpers](simulations/OT/helpers.py#L4-L42).

| Method | Implementation | Hyperparameters |
|---|---|---|
| Cocycle | Identity anchor at treatment 0. Treatments 1 and 2 each use a separate 2D monotonic-affine MAF with one autoregressive transform and a two-hidden-layer, width-32 conditioner; CMMD-U | One architecture (no architecture search); random permutation, fixed 50:50 validation, full-data retraining; Adam, LR 1e-2, 1,000 epochs, batch 128 ([runner](simulations/OT/run_cocycles.py#L31-L89)) |
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

| Estimator | Exact architecture search | Base and objective | Model selection and optimization |
|---|---|---|---|
| Cocycle | Conditional (d-1)-dimensional MAF for X2:d given scalar X1. Search A0 linear shift, A1 two-layer 32-wide neural shift, A2 two-layer 32-wide neural affine, and A3 three-transform affine–8-bin-RQS–affine with two 32-wide hidden layers per transform. | No latent base; CMMD-V with median-heuristic Gaussian output kernel | Two complementary folds; full-data retraining; Adam; 1,000 epochs per fold and retraining; batch 128; LR 1e-2 for A0–A2 and 1e-3 for A3; weight decay 0; no scheduler ([implementation](simulations/Csuite/run_cocycles_mixed.py#L140-L209)) |
| CAREFL | Joint d-dimensional autoregressive flow with no external context. Search A0 linear shift, A1 two-layer 32-wide neural shift, and A2 two-layer 32-wide neural affine. The wrapper's <code>affine=True</code> removes A3. | Three separate fits, using learnable factorized Normal, Laplace, or Student-t bases; NLL. The runner does not select among bases. | Shuffled two-fold selection within each base; full-data retraining; Adam; 1,000 epochs; batch 128; CV LR 1e-2 for A0–A1 and 1e-3 for A2 because the selector assigns the reduced search's last candidate the smaller rate; final retraining LR 1e-2; weight decay 0; no scheduler ([wrapper](simulations/Csuite/run_simcsuite_carefl.py#L6-L34), [implementation](simulations/Csuite/run_flows_mixed.py#L158-L236)) |
| CausalNF | Joint d-dimensional autoregressive flow with no external context. Search the full A0–A3 set, including the three-transform 8-bin spline MAF A3. | Three separate learnable factorized bases: Normal, Laplace, and Student-t; NLL | Shuffled two-fold selection within each base; full-data retraining; Adam; 1,000 epochs; batch 128; CV LR 1e-2 for A0–A2 and 1e-3 for A3; final retraining LR 1e-2; weight decay 0; no scheduler ([wrapper](simulations/Csuite/run_simcsuite_causalnf.py#L6-L34)) |
| BGM | Coupling flow on the full vector: leave X1 unchanged and conditionally transform X2:d. The inner outcome MAF searches the full A0–A3 set with the same two-layer, width-32 networks and 8-bin A3 spline as the cocycle. | Three separate learnable factorized bases: Normal, Laplace, and Student-t; NLL | Same shuffled two-fold likelihood selection and optimizer budget as CausalNF ([implementation](simulations/Csuite/run_bgm_mixed.py#L159-L238)) |

All methods report coordinatewise KS/W1 paired-change and interventional metrics plus paired counterfactual RMSE, as defined under Experiment 8.1.

## 401(k) application

| Component | Implemented value | Source |
|---|---|---|
| Data | <code>fetch_401K(return_type='DataFrame')</code>; saved run has 9,915 records and 3,682 eligible | [notebook](applications/e401k-Cocycles-NF.ipynb) |
| Treatment/outcome | <code>e401</code> / <code>net_tfa</code> | [notebook](applications/e401k-Cocycles-NF.ipynb) |
| Model inputs | age, inc, educ, fsize, marr, twoearn, db, pira, hown, plus e401 moved first | [notebook](applications/e401k-Cocycles-NF.ipynb) |
| Scaling | Divide non-binary inputs and the outcome by their sample SD; binary inputs unchanged; no centering | [notebook](applications/e401k-Cocycles-NF.ipynb) |
| Data use | Random permutation, then 100% used for selection and fitting; no held-out test set | [notebook](applications/e401k-Cocycles-NF.ipynb) |
| Model | Scalar-outcome conditional MAF with all 10 inputs as context. Search A0 linear shift, A1 two-layer 32-wide neural shift, A2 two-layer 32-wide neural affine, and A3 affine–8-bin-RQS–affine with two 32-wide hidden layers per transform; CMMD-V | [notebook](applications/e401k-Cocycles-NF.ipynb), [architecture](applications/architectures.py#L52-L133) |
| Training | Two complementary folds and full-data retraining; Adam; 1,000 epochs in each fit; batch 128; initial LR 1e-2 for A0–A2 and 1e-3 for A3; weight decay 1e-3; StepLR every epoch with multiplier 0.9 | [config](applications/e401k_cocycle_config.py#L1-L13) |
| Effects | Set treatment to 0 and 1 for every unit; evaluate relative to observed treatment/outcome; rescale by outcome SD. ATE uses all units and ATT observed treated units. | [notebook](applications/e401k-Cocycles-NF.ipynb) |
| Conditional summaries | Nadaraya–Watson regression on rank income or rank predicted Y(0); Gaussian kernel initialized at length scale 1, regularization 0; 5 folds, 1,000 iterations, LR 0.1, subsample 256; 500 prediction points on [0.1,0.9] | [notebook](applications/e401k-Cocycles-NF.ipynb) |

## Software environment

The supplied environment fixes Python 3.12.2, NumPy 1.26, PyTorch 2.1.2, Matplotlib 3.8.3, pandas 2.2.0, DoubleML 0.7.1, causalflows 0.1.0, Zuko 1.4.0, and seaborn 0.13.2 ([environment](environment.yml)).

## Current-code verification notes

1. Cocycle runner arguments named <code>k_folds</code> are not passed to <code>validate</code>; the 0.5 split fixes the actual count at two ([8.1](simulations/linear_model/run_cocycles.py#L153-L164), [8.3](simulations/Csuite/run_cocycles_mixed.py#L140-L149)).
2. The Experiment 8.2 builder comment refers to four architectures, but the current factory returns one anchored affine-MAF candidate ([builder](simulations/OT/run_cocycles.py#L11-L20), [factory](simulations/OT/architectures.py#L235-L263)).
3. Likelihood-flow CV assigns LR 1e-3 to the last architecture in the supplied search (A3 in the full search and A2 in CAREFL's reduced search), but final retraining uses the base LR 1e-2 for whichever candidate wins ([selector](causal_cocycle/causalflow_helper.py#L137-L173)).
4. Local Experiment 8.2 launchers use correct coordinate order; reversed-order ablations are in the HPC launchers.
5. The 401(k) notebook does not seed its random permutation. Its later quantile cell refers to <code>NWConditioner</code> and <code>KREpsLayer</code> without importing them in the notebook, so those plotting cells require the classes in the active kernel namespace.
6. The 401(k) quantile computation requests (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975), while the plot labels these curves as (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95). Cocycle fitting, ATE, and ATT precede and do not depend on this plotting-label mismatch.
7. In Design I Seq-OT, <code>torch.column_stack((Y0,Y1,Y2))</code> creates six columns rather than pooling rows. The subsequent conditional-map code uses only columns 0 and 1, so the implementation behavior is the control-arm construction stated in the method table.
