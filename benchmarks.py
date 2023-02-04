from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp


from helper_funcs import * 
from pympc.geometry.polyhedron import Polyhedron
from pympc.dynamics.discrete_time_systems import LinearSystem, PieceWiseAffineSystem
from pympc.control.controllers import ModelPredictiveController
from pympc.plot import plot_input_sequence, plot_state_trajectory, plot_state_space_trajectory

from sampling import sample_polyhedron


class Benchmark(Formulated):
    def __init__(self) -> None:
        super().__init__()
        self.dimension = 2
        self.lb : np.array = None
        self.ub : np.array = None
        
    def f(self, x): pass
    def torch_formulation(self, x): pass
    def gurobi_formulation(self, model:gp.Model, x): pass
    def sample_in_domain(self, batch_size, x=None): pass
    def in_domain(self, x): pass
    
class PWA(Benchmark):
    """
    Piecewise Affine system derived from MPC controller.
    Example from https://arxiv.org/pdf/2012.12015.pdf section 5.1
    """
    def __init__(self) -> None:
        super().__init__()
        self.dimension = 2
        self.lb = np.array([-5., -5.])
        self.ub = np.array([5., 5.])
        self.A = np.array([
            [1.2, 1.2],
            [0, 1.2]
        ])

        self.B = np.array([
            [1.],
            [0.5]
        ])
        
        S = LinearSystem(self.A, self.B)
        Q = np.eye(2) * 10
        R = np.eye(1)

        P, K = S.solve_dare(Q, R)

        umin, umax = np.array([-1.]), np.array([1.])
        xmin, xmax = np.array([-5., -5]), np.array([5., 5])
        U = Polyhedron.from_bounds(umin, umax)
        X = Polyhedron.from_bounds(xmin, xmax)
        D = X.cartesian_product(U)

        X_N = S.mcais(K,D)
        self.controller = ModelPredictiveController(S, N=10, Q=Q, R=R, P=P, D=D, X_N=X_N)
        
        print("PyMPC: Computing critical regions ... ")
        self.controller.store_explicit_solution(verbose=False)
        self.invariant_set = self.controller.mpqp.get_feasible_set()

        
        self.modes = []
        for cr in self.controller.explicit_solution.critical_regions:
            poly = cr.polyhedron
            poly.remove_redundant_inequalities()
            self.modes.append((poly, cr._u['x'][0], cr._u['0'][0]))
    
    def f(self, x):
        u = self.controller.feedback_explicit(x)
        # print(x.shape)
        # print(u.shape)
        # self.A @ x
        # self.B @ u
        return self.A @ x + self.B @ u
    
    def gurobi_formulation(self, model:gp.Model, x):
        """
        :param model: gp.Model
        :param x: gp.MVar, representing the input of dynamical system
        :return:
            gp.MVar, representing the output of dynamical system
        """
        xs = []
        mus = grb_bin_var( model, (len(self.modes),) )
        uacc = []
        for i, (poly, ux, u1) in enumerate(self.modes):
            xi = grb_cont_var( model, (self.dimension,) ) 
            
            model.addConstr(poly.A @ xi <= mus[i] * (poly.b - 1e-3))
            if poly.C.shape[0] > 0:
                model.addConstr(poly.C @ xi == mus[i] * poly.d)
            uacc.append( ux @ xi + mus[i] * u1 )
            xs.append(xi)
        uacc = grb_sum(uacc)
        
        model.addConstr( x == grb_sum(xs) )
        xp = grb_cont_var(model, (self.dimension,) )
        model.addConstr( xp == self.A @ x +  uacc * self.B[:,0] )
        model.addConstr( grb_sum(mus) == 1 )
        return xp

    
    def sample_in_domain(self, batch_size, x=None):
        return sample_polyhedron(self.invariant_set, batch_size)

    def in_domain(self, x):
        return self.invariant_set.contains(x)

    
        
if __name__ == '__main__':
    pwa = PWA()
    pwa.controller.plot_state_space_partition()
    x = np.array([3.02241846, 1.05304048])
    plt.scatter(x[0], x[1])
    x = np.array([2.60337363, 1.13827178])
    plt.scatter(x[0], x[1])
    x = np.array([-1.3957, -1.3678])
    plt.scatter(x[0], x[1])
    plt.show()



    