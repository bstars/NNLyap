import unittest
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import cvxpy

from benchmarks import PWA
from sampling import sample_polyhedron
from neural import ReluFCNet
from config import Config
from helper_funcs import *

pwa = PWA()

class Test(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.pwa = pwa
        self.net = ReluFCNet([2, 5, 3])

    def test_pwa_formulation(self):
        """
        test the piecewise-affine formulation for dynamical system
        make sure that the output matches with the input
        """
        xs = self.pwa.sample_in_domain(5)

        for i in range(len(xs)):
            truth = self.pwa.f(xs[i])

            model = grb_model()
            model.Params.LogToConsole = 0
            x = grb_cont_var(model, (self.pwa.dimension, ))
            xp = self.pwa.gurobi_formulation(model, x)
            model.addConstr( x == xs[i] )
            model.setObjective(0, GRB.MINIMIZE)
            model.optimize()

            self.assertTrue(
                np.all( np.abs(truth - xp.X) <= 1e-5 )
            )

    def test_fcnet_formulation(self):
        """
       test the MIP formulation for ReLU neural net
       make sure that the output matches with the input
       """
        x_val = self.pwa.sample_in_domain(1)[0]
        phix_val = self.net(array_to_tensor(x_val[None, :]))[0].cpu().detach().numpy()

        model = grb_model(convex=True)
        x = grb_cont_var(model, (2,))
        [phix] = self.net.gurobi_formulation(model, [x], -5 * np.ones([2]), 5 * np.ones([2]))
        model.addConstr(x == x_val)
        model.setObjective(0, GRB.MINIMIZE)
        model.optimize()

        print(phix.X - phix_val)
        self.assertTrue(
            np.all(np.abs(phix.X - phix_val) <= 1e-4)
        )

    def test_mip_formulation(self):
        """
        Test the MIP formulation for composition of dynamical system and ReLU neural net
        """
        x_val = self.pwa.sample_in_domain(1)[0]
        xp_val = self.pwa.f(x_val)
        phixp_val = self.net( array_to_tensor(xp_val[None,:]) )[0].cpu().detach().numpy()

        model = grb_model(convex=True)
        x = grb_cont_var(model, (2,))
        xp = self.pwa.gurobi_formulation(model, x)
        [phixp] = self.net.gurobi_formulation( model, [xp], -5 * np.ones([2]), 5 * np.ones([2]) )
        model.addConstr(x == x_val)

        model.setObjective(0, GRB.MINIMIZE)
        model.optimize()

        print(phixp.X - phixp_val)
        self.assertTrue(
            np.all( np.abs(phixp.X - phixp_val) <= 1e-4 )
        )





if __name__ == '__main__':
    unittest.main()

    # pwa = PWA()
    # model = grb_model()
    # x = grb_cont_var(model, (2,))
    # y = pwa.gurobi_formulation(model, x)
    # model.addConstr(x == np.array([3.02241846, 1.05304048]))
    #
    # model.setObjective(0, GRB.MINIMIZE)
    # model.optimize()

    