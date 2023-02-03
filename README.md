# Hierarchical Bayesian Model
A Hierarchical Bayesian Model containing a trial-by-trial learning update parameter, alpha. Alpha can take the form of a polynomial (HBM_main_sims_polynomial.py) or sigmoid (HBM_main_sims_sigmoid.py). 

MaximumAPosterioriModel.py contains the model, which encodes observations and latent random variables, and the guide, which defines the variational distribution.

SetParameters.py contains the set of regression equations for calculation of parameter estimates that evolve stochastically from trial to trial. 
