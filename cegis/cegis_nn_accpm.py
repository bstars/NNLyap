"""
Formulate the Lyapunov function as V(x) = || L \phi(x)  ||_2^2 = \phi(x)^T L^T L \phi(x)

First train \phi with sampled data points, then run cegis with ACCPM on P = L^T L
"""


import sys
sys.path.append('..')


import torch
from torch import nn 
import cvxpy
import matplotlib.pyplot as plt

from neural import ReluFCNet
from benchmarks import PWA, Benchmark
from helper_funcs import * 

class Learner():
    def __init__(self, n, alpha=1e-5, beta=1.) -> None:
        """__init__ _summary_

        :param alpha: _description_, defaults to 1e-5
        :param beta: _description_, defaults to 1.
        """

        self.P = cvxpy.Variable([n, n], symmetric=True)
        self.alphaI = np.eye(n) * alpha
        self.betaI = np.eye(n) * beta
        self.obj = cvxpy.log_det(self.betaI - self.P) + cvxpy.log_det(self.P - self.alphaI)

    def learn(self):
        print_segline('Learner')
        prob = cvxpy.Problem(
            cvxpy.Minimize(-self.obj)
        )
        prob.solve(Config.AC_SOLVER)
        assert prob.value is not None
        assert prob.value < np.inf
        return self.P.value
    
    def augment(self, phix, phixp):
        norm = np.linalg.norm(
            np.outer(phixp, phixp) - np.outer(phix, phix),
            ord='fro'
        )
        
        self.obj = self.obj + cvxpy.log(
            1 / norm * phix @ self.P @ phix
            - 1 / norm * phixp @ self.P @ phixp
        )
    
class Verifier():
    def __init__(self, benchmark : Benchmark, net : ReluFCNet, eps=1e-1) -> None:
        
        self.benchmark = benchmark
        self.net = net
        self.dimension = self.benchmark.dimension
        
        self.model = grb_model('Verifier')
        self.model.params.NonConvex = 2
        
        # build the computation graph
        """ 
        x -> [ Dynamical system ] -> xp
        |                            |
        V____________     ___________V
                    |     |
                    |     |
                    V     V
                   [net phi]
                    |     |
                    |     |
                    V     V
                phi(x)    phi(xp)
        """
        self.x = grb_cont_var(self.model, (self.benchmark.dimension, ))
        xp = self.benchmark.gurobi_formulation(self.model, self.x)
        [self.phix, self.phixp] = self.net.gurobi_formulation(
            self.model, [self.x, xp], self.benchmark.lb, self.benchmark.ub
        )
        
        # constriants ||x||_inf >= eps
        eps_vars = [grb_bin_var(self.model, (self.dimension,)) for _ in range(2)]
        self.model.addConstr( self.x <= -eps + 10 * eps_vars[0] )
        self.model.addConstr( self.x >= eps - 10 * eps_vars[1] )
        self.model.addConstr( grb_sum(eps_vars[0]) + grb_sum(eps_vars[1]) <= (2 * self.dimension - 1) )
        
    def verify(self, P):
        print_segline('Verifier')
        self.model.setObjective(
            self.phixp @ P @ self.phixp - self.phix @ P @ self.phix, GRB.MAXIMIZE
        )
        self.model.optimize()
        obj = self.model.ObjVal
        
        if obj > -1e-8:
            return False, self.x.X
        else:
            return True, None
        
class Cegis():
    def __init__(self, benchmark : Benchmark, dims) -> None:
        assert dims[0] == benchmark.dimension
        self.benchmark = benchmark
        self.net = ReluFCNet(dims=dims[:-1])
        self.L = nn.Parameter( torch.randn(dims[-1], dims[-2]) )

        self.learner = Learner(dims[-2])
        self.verifier = Verifier(self.benchmark, self.net)
        
        
    def V(self, x):
        x = array_to_tensor(x)
        phix =  self.L @ self.net(x).T # [dim, batch]
        return torch.sum(torch.square(phix), dim=0)
        
    
    def pretrain(self, num_pretrain=30, max_iterations = 100000):
        xs = self.benchmark.sample_in_domain(batch_size=num_pretrain)
        xps = np.array([
            self.benchmark.f(x) for x in xs
        ])
        
        xs = array_to_tensor(xs)
        xps = array_to_tensor(xps)
        
        optimizer = torch.optim.Adam(params= list(self.net.parameters()) + [self.L], lr=1e-3)
        
        print('Pretraining neural net')
        for _ in range(max_iterations):
            
            optimizer.zero_grad()
            
            loss = torch.sum(
                torch.relu(
                    self.V(xps) - self.V(xs) + 1
                )
            )
            
            loss.backward()
            optimizer.step()
            
            print(loss.item())
            
            if loss == 0.:
                return 
        raise Exception("Cannot satisfy decreasing condition on sampled data")

    
    def solve(self):
        L = self.L.detach().numpy()
        P = L.T @ L
        while True:
            verified, ctx = self.verifier.verify(P)
            if verified:
                
                return P, self.net
            print("Counter example found:")
            print(ctx, ctx.shape)
            phix = self.net( array_to_tensor(ctx[None, :]) )[0].detach().numpy()
            phixp = self.net( array_to_tensor( self.benchmark.f(ctx)[None, :] ) )[0].detach().numpy()
            self.learner.augment(phix, phixp)
            P = self.learner.learn()
            

if __name__ == '__main__':
    pwa = PWA()
    cegis = Cegis(pwa, [2, 10, 5, 5])
    cegis.pretrain(num_pretrain=100)

    P, net = cegis.solve()

    x = np.linspace(-5, 5, 1000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    xys = np.stack([X, Y], axis=-1).reshape([-1, 2])
    phix = net(array_to_tensor(xys)).detach().numpy()
    vals = np.sum((phix @ P) * phix, axis=1)
    vals = vals.reshape([len(x), len(x)])

    plt.contourf(X, Y, vals)
    plt.colorbar()
    plt.show()
    


    
    
    
        
    
    