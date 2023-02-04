
from random import sample
import sys
from unicodedata import name
sys.path.append("..")

import numpy as np
import cvxpy
import matplotlib.pyplot as plt


from pympc.geometry.polyhedron import Polyhedron
from config import Config

def sample_unit_circle(size):
    """sample_unit_circle 
    
    Uniformly sample on a unit circle
    
    :param size: 
    :return: 
    """
    m, n = size
    samples = np.random.randn(m, n)
    norms = np.linalg.norm(samples, axis=1)[:,None]
    return samples / norms
    


def sample_polyhedron(poly:Polyhedron, size, ac=None, warm_up=20):
    """sample_polyhedron _summary_
    
    Sample uniformly in a polyhedron
    
    :param poly: _description_
    """
    A, b, C, d = poly.A, poly.b, poly.C, poly.d
    m, n = A.shape

    
    if ac is None:
        x = cvxpy.Variable([n])
        prob = cvxpy.Problem(
            cvxpy.Minimize( -cvxpy.sum(cvxpy.log(b - A @ x)) ),
            constraints=[] if C.shape[0]==0 else [C @ x == d]
        )
        prob.solve('ECOS')
        x = x.value
    else:
        x = ac
        
    ret = []
    u = cvxpy.Variable()
    for _ in range(warm_up + size):
        direction = sample_unit_circle([1,2])[0]
        prob1 = cvxpy.Problem(
            cvxpy.Minimize(u),
            constraints = [A @ (x + u * direction) <= b] + [] if C.shape[0]==0 else [C @ (x + u * direction) == d]
        ) 
        prob1.solve(Config.LP_SOLVER)
        low = u.value
        
        prob2 = cvxpy.Problem(
            cvxpy.Maximize(u),
            constraints = [A @ (x + u * direction) <= b] + [] if C.shape[0]==0 else [C @ (x + u * direction) == d]
        ) 
        prob2.solve(Config.LP_SOLVER)
        high = u.value
        
        # print(low, high)
        
        u1 = np.random.uniform(low=low, high=high)
        x = x + u1 * direction
        if np.linalg.norm(x) >= 1e-1:
            ret.append(x)
    
    return np.array(ret)[-size:, :]
        
         
    
def eg_sample_unit_circle():
    samples = sample_unit_circle([100, 2])
    plt.scatter(samples[:,0], samples[:,1])
    plt.show()

def eg_sample_poly():
    A = np.array([
        [-1, 1],
        [1, -1],
        [-1, -1],
        [1, 1.]
    ])
    b = np.ones([4])
    poly = Polyhedron(A, b)
    samples = sample_polyhedron(poly, size=300)
    plt.scatter(samples[:,0], samples[:,1], s=6)
    
    x = np.linspace(0, 1, 100)
    plt.plot(x, x - 1)
    plt.plot(x, -x + 1)
    
    x = np.linspace(-1, 0, 100)
    plt.plot(x, x + 1)
    plt.plot(x, -x - 1)
    
    plt.show()
    
if __name__ == '__main__':
    eg_sample_poly()