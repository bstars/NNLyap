"""
Formulate the Lyapunov function as V(x) = || L \phi(x)  ||_2^2 = \phi(x)^T L^T L \phi(x)

First train \phi with sampled data points, then run cegis with SGD
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


class Cegis():
	def __init__(self, benchmark : Benchmark, dims):
		assert dims[0] == benchmark.dimension
		self.benchmark = benchmark
		self.net = ReluFCNet(dims=dims[:-1])
		self.L = nn.Parameter(torch.randn(dims[-1], dims[-2]))

		xs = self.benchmark.sample_in_domain(batch_size=500)
		xps = np.array([
			self.benchmark.f(x) for x in xs
		])

		self.xs = array_to_tensor(xs)
		self.xps = array_to_tensor(xps)

	def V(self, x):
		x = array_to_tensor(x)
		phix = self.L @ self.net(x).T  # [dim, batch]
		v = torch.sum(torch.square(phix), dim=0)
		# print(v.shape)
		return v

	def augment(self, x):
		xp = array_to_tensor(self.benchmark.f(x))
		x = array_to_tensor(x)

		self.xs = torch.cat([self.xs, x[None, :]], dim=0)
		self.xps = torch.cat([self.xps, xp[None, :]], dim=0)


	def learn(self, max_iterations=50000):
		print_segline("Learner")
		print("%d examples" % (self.xs.shape[0]))
		optimizer = torch.optim.Adam(params=list(self.net.parameters()) + [self.L], lr=1e-3)

		for _ in range(max_iterations):

			optimizer.zero_grad()

			diffs = self.V(self.xps) - self.V(self.xs)

			loss = torch.sum(
				torch.relu( diffs + 0.01 )
			)

			loss.backward()
			optimizer.step()
			print(loss.item())

			if loss == 0.:
				print(self.xs[-1,:], diffs[-1])
				return
		raise Exception("Cannot satisfy decreasing condition on sampled data")

	def verify(self, eps=1e-5):
		"""
		Since we are training the neural net every time, we need to compute the MIP formulation of NN every time
		:param eps: The epsilon ball parameter
		"""
		print_segline("Verifier")
		P = (self.L.T @ self.L).cpu().detach().numpy()
		model = grb_model(convex=False)
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
		x = grb_cont_var(model, (self.benchmark.dimension,))
		xp = self.benchmark.gurobi_formulation(model, x)
		[phix, phixp] = self.net.gurobi_formulation(
			model, [x, xp], self.benchmark.lb, self.benchmark.ub
		)

		# constriants ||x||_inf >= eps
		M = self.benchmark.ub - self.benchmark.lb
		eps_vars = [grb_bin_var(model, (self.benchmark.dimension, )) for _ in range(2)]
		model.addConstr(x <= -eps + M * eps_vars[0])
		model.addConstr(x >= eps - M * eps_vars[1])
		model.addConstr(grb_sum(eps_vars[0]) + grb_sum(eps_vars[1]) <= (2 * self.benchmark.dimension - 1))

		model.setObjective(
			phixp @ P @ phixp - phix @ P @ phix, GRB.MAXIMIZE
		)

		model.optimize()
		obj = model.ObjVal

		if obj > -1e-8:
			print("Counter example found:", x.X)
			print("Violation: %.3f" % (obj))
			return False, x.X, P
		else:
			return True, None, P

	def solve(self):
		while True:
			self.learn()
			verified, ctx, P = self.verify()
			if verified:
				return P, self.net
			self.augment(ctx)

if __name__ == '__main__':
	pwa = PWA()
	cegis = Cegis(pwa, [2, 10, 10, 10, 10])
	try:
		cegis.learn(max_iterations=50000)
		# cegis.solve()
	except:
		print("Cannot satisfy decreasing condition on sampled data, plotting ....")
		x = np.linspace(-5, 5, 3000)
		y = np.linspace(-5, 5, 3000)
		X, Y = np.meshgrid(x, y)
		xys = np.stack([X, Y], axis=-1).reshape([-1, 2])
		vals = cegis.V(xys).cpu().detach().numpy()

		for i in range(len(vals)):
			if not pwa.in_domain(xys[i]):
				vals[i] = np.nan

		fig = plt.figure()
		ax = plt.axes()
		# ax = plt.axes(projection='3d')

		im = ax.contourf(X, Y, np.reshape(vals, [len(x), len(y)]))
		ax.scatter( cegis.xs[:,0].cpu().detach().numpy(),  cegis.xs[:,1].cpu().detach().numpy(), s=2, c='red')
		fig.colorbar(im)
		plt.show()
