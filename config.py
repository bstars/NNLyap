import torch

class Config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    LP_SOLVER = "ECOS"
    MIP_SOLVER = 'GUROBI'
    AC_SOLVER = "MOSEK"
    
    GUROBI_VERBOSE = False
    