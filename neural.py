import cvxpy
import torch
from torch import nn
import numpy as np
import gurobipy as gp

from helper_funcs import * 


class ReluFCNet(nn.Module, Formulated):
    """ReluFCNet 
    Fully-connected neural net with relu activation for hidden layers, no activation for final output.
    This Class should not be used directly. Use LYNet and DNet instead
    :param nn: _description_
    """
    def __init__(self, dims, bias=False, positive_w_final = False) -> None:
        super().__init__()
        layers = []

        for i in range(len(dims)-2):
            layers.append( nn.Linear(dims[i], dims[i+1], bias=bias) )
            layers.append( nn.ReLU() )
            # layers.append( nn.LeakyReLU() )
            
        linear = nn.Linear(dims[-2], dims[-1], bias=bias)
        if positive_w_final:
            linear.weight.data.uniform_()
        layers.append( linear )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def get_param_pair(self):
        """get_param_pair 


        :return: 
            ws: Weight variables in relu network
            bs: Bias variables in relu network
            ws and bs must have the same length

        """
        ws = []
        bs = []
        
        for name, param in self.layers.named_parameters():

            if "weight" in name: 
                # print(param.shape)
                ws.append(param.detach().numpy())
            elif "bias" in name: 
                # print(param.shape)
                bs.append(param.detach().numpy())
        if len(bs) == 0:
            bs = [
                np.zeros([ w.shape[0] ]) for w in ws
            ]
        return ws, bs
    
    def gurobi_formulation(self, model:gp.Model, xs, l, u):
        """gurobi_formulation _summary_

        :param model: gurobipy.Model
        :param xs: A list of gp.MVar, indicating the inputs of neural network
        :param l: Lower bounds of input
        :param u: Upper bounds of input
        """
        n_input = len(xs)
        ws, bs = self.get_param_pair()
        

        (Ls, Us) = neural_net_bounds(ws, bs, l, u)
        for x in xs:
            model.addConstr(x <= u)
            model.addConstr(x >= l)
            
        for i, (w,b) in enumerate(zip(ws, bs)):
            m, n = w.shape
            
            ys = []

            if i < len(ws)-1:
                for j in range(n_input):
                        y = grb_cont_var(model, (m,))
                        z = grb_bin_var(model, (m,))
                        model.addConstr( y >= w @ xs[j] + b )
                        model.addConstr( y >= 0 )
                        model.addConstr( y <= w @ xs[j] + b - (1 - z) * Ls[i+1] )
                        model.addConstr( y <= z * Us[i+1] )
                        ys.append(y)
            else:
                for j in range(n_input):
                    y = grb_cont_var(model, (m,))
                    model.addConstr(y == w @ xs[j] + b)
                    ys.append(y)

            xs = ys
        return ys
    
  
if __name__ == '__main__':
    net = ReluFCNet([2, 10, 10, 5])
    w = nn.Parameter(torch.randn(10))
    print(
        type(
            list(net.parameters()) + [w]
        )
    )