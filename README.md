This repository contains all Python source code used for all market simulations, and results from simulations can be found in [here](/AllResults/analysis/).

- [`BSE.py`](BSE.py) used for all simulations except GP Simulations - contains all the functions that are central to market configuration and the Exchange/LOB 
- [`BSE_GP.py`](BSE_GP.py) used for GP simulations - contains the code for the evolved trader 
- [`traders.py`](traders.py) contains code for all the traders used in this project 
- [`op_models.py`](op_models.py) contains the code for Opinion Dynamics models reused from https://github.com/ken-neth/opinion_dynamics_BSE.
- [`PMF_joins.py`](PMF_joins.py) contains the code for the functions used to implement PMF composition mechanisms; note, in `traders.py` the `O-PRZI-GP` trader has these composition mechanisms built into it.
