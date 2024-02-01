## Bayesian Model for the Rewards, aka Medicare Not-Obviously Heat-Related (NOHR) Hospitalizations
*Goal: model the rewards such that (a) the coefficients are interpretable across space and time, (b) we can explicitly include domain knowledge, (c) uncertainty quantification is straightforward.* 

**Notation**

* Let $s\in\mathcal{S}$ and $t\in\mathcal{T}$ denote a county and time (in days), respectively.  
* Let $Y_{s,t}$ denote the number of NOHR hospitalizations.
* Let $A_{s,t}$ denote whether or not a heat alert was issued.
* Let $X_{s,t}$ denote the temporally-varying covariates at location $s$.
* Let $W_{s}$ denote the spatially-varying but temporally static covariates. 
* Let $\tau_{s,t}$ be the multiplicative effect of $a=1$.

**General Approach**

* A hierarchical model explaining $\tau_{s,t}$ as a function of the covariates, with the restriction that $\tau_{s,t} \in \[0,1\]$.
* Local dependence on $(s,t)$ is modulated by heat index and previous heat alerts issued.
* This is the "meta-analysis" approach, and it's meant to solve the spatial confounding problem (such that there is no spatial confounding at the local level). 

**Model**

To make the model scalable we need to assume linearity.

$$
\begin{aligned}
Y_{s,t} & \sim \text{Poisson}(N_{s,t} * \lambda_{s,t} * (1 - A_{s,t}\tau_{s,t})) \\
\lambda_{s,t} & = exp(\beta_{s}^\top X_{s,t}) \\
\tau_{s,t} & = exp(\gamma_{s}^\top V_{s,t}) \\
\end{aligned}
$$

where $N_{s,t}$ is the summer-specific mean of NOHR hospitalizations at location $s$, $\lambda_{s,t}$ is the baseline variation from the mean (with mean 1), and $V_{s,t}$ is a subset of $X_{s,t}$.

To make this model hierarchical, we specify priors depending on the spatial variables. To incorporate domain knowledge, we also impose constraints on some of the variables. Let $C_{bi}$ denote the constraint (or lack thereof) for variable $i \in \[1,length(\beta_s)\]$ and $C_{gi}$ denote the constraint (or lack thereof) for variable $i \in \[1,length(\gamma_s)\]$.

First:

$$
\begin{aligned}
b_{s} \sim MLP(W_s)\\
g_{s} \sim MLP(W_s)\\
\end{aligned}
$$

Next, for i in 1 to length($b_s$):
* if $C_{bi}$ == None: $\beta_{s}\[i\] \sim Normal(b_s\[i\],1)$
* else if $C_{bi}$ == "positive": $\beta_{s}\[i\] \sim LogNormal(b_s\[i\],1)$
* else if $C_{bi}$ == "negative": $\beta_{s}\[i\] \sim NegativeLogNormal(b_s\[i\],1)$

And for i in 1 to length($g_s$):
* if $C_{gi}$ == None: $\gamma_{s}\[i\] \sim Normal(g_s,1)$
* else if $C_{gi}$ == "positive": $\gamma_{s}\[i\] \sim LogNormal(g_s\[i\],1)$
* else if $C_{gi}$ == "negative": $\gamma_{s}\[i\] \sim NegativeLogNormal(g_s\[i\],1)$

**Model Fitting**

We use stochastic variational inference (SVI) with a low-rank multivariate normal distribution as the guide for the posterior. You can read more about estimating and optimizing the evidence lower bound (ELBO) with SVI using Pyro [here](http://pyro.ai/examples/intro_long.html#Background:-Estimating-and-optimizing-the-Evidence-Lower-Bound-(ELBO)).

For the Multi-Layer Perceptron (MLP) priors and for the overall model fitting, we use Pytorch / Pytorch Lightning.

