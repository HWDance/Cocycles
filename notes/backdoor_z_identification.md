# Back-Door `Z` Adjustment and Conditional Transport

This note summarizes the discussion around when adding `Z` as a conditioning covariate is enough to identify the `X`-dependent change in the outcome law.

## Setup

We are interested in counterfactual changes in `X`, not in intervening on `Z`.

The target object is therefore:

\[
P(Y(x') \mid Z=z)
\]

or, at the map level for scalar `Y`, the transport from the observed law under `X=x` to the counterfactual law under `X=x'`, while holding `Z` fixed.

## Key Condition

Conditioning on `Z` is sufficient if:

\[
Y(x) \perp X \mid Z
\]

or equivalently, in a structural model

\[
Y = g(X,Z,U), \qquad U \perp X \mid Z.
\]

Under this condition,

\[
P(Y \mid X=x, Z=z) = P(Y(x) \mid Z=z).
\]

This is the back-door identification statement. It says that after fixing `Z`, the observational conditional law matches the interventional conditional law for `X`.

## What Does Not Matter

It is **not** a problem that `Z` is correlated with the outcome noise.

For example, if

\[
Y = f(X,Z) + U
\]

and `Z` is correlated with `U`, this does **not** by itself invalidate adjustment for the effect of `X`. What matters is whether

\[
U \perp X \mid Z.
\]

So:

- `Z` may be correlated with the noise in `Y`
- the effect of `X` can still be identified by conditioning on `Z`
- the causal effect of `Z` itself is generally **not** identified from this representation alone

## What Goes Wrong Without Conditioning

If

\[
Y = X + \xi, \qquad \xi \not\perp X,
\]

then the observational conditional law

\[
P(Y \mid X=x)
\]

is not equal to the interventional law

\[
P(Y(x)).
\]

So a transport or KR map built from `P(Y|X=x)` to `P(Y|X=x')` is biased for the counterfactual problem.

The same logic applies conditionally:

\[
P(Y \mid X=x, Z=z)
\]

is valid for counterfactual transport **only if**

\[
Y(x) \perp X \mid Z.
\]

If `Z` is only a proxy and does not fully block confounding, then even the conditional law remains biased.

## Scalar `Y`: Unique Monotone Map

When `Y` is one-dimensional and the back-door condition holds, then for each fixed `z` the causal transport from `x` to `x'` is the unique monotone increasing map

\[
T_{x \to x'}^{z}(y)
=
F_{Y \mid X=x', Z=z}^{-1}\!\left(F_{Y \mid X=x, Z=z}(y)\right).
\]

This is the clean scalar-outcome theory:

1. identify the conditional laws `P(Y|X=x,Z=z)` and `P(Y|X=x',Z=z)`
2. build the monotone transport between them
3. apply it at the observed `z`

In this setting, adding `Z` as a conditioning covariate is enough to identify the `X`-dependent change, provided conditional ignorability holds.

## Why We Hold `Z` Fixed

One can define a more general family of transports

\[
T_{x,x',z,z'}
:
P(Y \mid X=x', Z=z') \to P(Y \mid X=x, Z=z),
\]

but for the back-door problem the causal object is only the diagonal slice

\[
T_{x,x',z,z}.
\]

That is because the counterfactual changes `X` while holding the confounder `Z` fixed.

So:

- off-diagonal transports with `z \neq z'` may exist
- they are not the causal estimand here
- the causal transport is the within-`Z` transport

## Implication for the Experiment

For the back-door experiment:

- `C = Z` is the CME conditioning set
- cocycle `Z` is empty
- seqOT/OT with conditioning on `Z` is targeting the correct conditional counterfactual law if the DGP satisfies `Y(x) \perp X \mid Z`

In the current back-door DGP, that condition is designed to hold, so conditional-on-`Z` transport is a valid identification strategy for the effect of `X`.
