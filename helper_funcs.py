from functools import reduce
import cvxpy
import torch
from torch import nn
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from colorama import Fore, Back, Style

from config import Config



class Formulated():
    def gurobi_formulation(self, model, x): pass
    
def print_segline(title=""):
    print(Fore.RED + "------------------------------------------------", end='')
    print(title, end='')
    print("------------------------------------------------")
    print(Style.RESET_ALL)
    
""" Helper functions for CVXPY formulation """

def bounds_lp(w, b, l, u):
    """bounds_lp 
    Solve the two optimization problems

        min_x.  <w,x> + b       max_x.  <w,x> + b
        s.t.    l <= x <= u     s.t.    l <= x <= u

    :param w <np.array> : Shape [n]  
    :param b <np.array> : Shape [n]  
    :param l <np.array> : Shape [n]  
    :param u <np.array> : Shape [n]  
    """
    wpos = w >= 0
    wneg = w < 0
    p1 = np.sum(w[wpos] * l[wpos]) + np.sum(w[wneg] * u[wneg]) + b
    p2 = np.sum(w[wpos] * u[wpos]) + np.sum(w[wneg] * l[wneg]) + b
    return p1, p2

def neural_net_bounds(ws, bs, l, u):
    """neural_net_bounds
    Compute all pre-activation bounds in relu network

    :param ws: 
    :param bs:
    :param l <np.array> : Shape [n]
    :param u <np.array> : Shape [n]
    :return
        Ls: Pre-activation lower bounds for each layer of relu net
        Us: Pre-activation upper bounds for each layer of relu net
    """
    Ls = [l]
    Us = [u]
    for w, b in zip(ws, bs):
        m, n = w.shape
        newl, newu = [], []

        for i in range(m):
            p1, p2 = bounds_lp(w[i,:], b[i], l, u)
            newl.append(p1)
            newu.append(p2)
        newl = np.array(newl); newu = np.array(newu)
        Ls.append(newl); Us.append(newu)

        l = np.maximum(newl, 0); u = np.maximum(newu, 0)
    return Ls, Us

""" PyTorch helper functions """
def array_to_tensor(x): 
    return torch.Tensor(x).to(Config.DEVICE)


""" GUROBI helper functions """
def grb_cont_var(model:gp.Model, shape):
    return model.addMVar(shape, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)

def grb_bin_var(model:gp.Model, shape):
    return model.addMVar(shape, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.BINARY)

def grb_sum(x):
    if type(x) == list:
        return reduce( lambda x, y : x+y, x)
    
    if type(x) == gp.MVar:
        return gp.quicksum(x[i] for i in range(x.shape[0]))

def grb_concatenate(model : gp.Model, vars):
    """ Inefficient function that add lots of constraints """
    n = sum( var.shape[0] for var in vars )
    y = grb_cont_var(model, (n,))
    nsum = 0
    for v in vars:
        model.addConstr(y[nsum : nsum + v.shape[0]] == v)
        nsum += v.shape[0]
    return y

def grb_model(convex=True):
    m = gp.Model()
    m.Params.LogToConsole = Config.GUROBI_VERBOSE
    if not convex:
        m.params.NonConvex = 2
    return m

if __name__ == '__main__':
    model = grb_model()
    x1 = grb_cont_var(model, (2,))
    x2 = grb_cont_var(model, (2,))
    x3 = grb_cont_var(model, (2,))
    y = grb_concatenate(model, [x1, x2, x3])

    model.addConstr(x1 == np.array([1., 1.]))
    model.addConstr(x2 == np.array([1., 1.]))
    model.addConstr(x3 == np.array([2., 2.]))
    model.setObjective(0)
    model.optimize()
    print(y.X)



