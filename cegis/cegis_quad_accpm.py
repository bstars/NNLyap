"""
Learning Lyapunov Functions for Hybrid Systems
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
	def __init__(self, benchmark:Benchmark, order=2, alpha=1e-5, beta=1.):
		n = benchmark.dimension * (order + 1)
		self.order = order
		self.benchmark = benchmark
		self.P = cvxpy.Variable([n, n], symmetric=True)
		self.alphaI = np.eye(n) * alpha
		self.betaI = np.eye(n) * beta
		self.obj = cvxpy.log_det(self.betaI - self.P) + cvxpy.log_det(self.P - self.alphaI)

	def learn(self):
		prob = cvxpy.Problem(
			cvxpy.Minimize(-self.obj)
		)
		prob.solve(Config.AC_SOLVER)
		assert prob.value is not None
		assert prob.value < np.inf
		return self.P.value

	def augment(self, x):
		xs = [x]
		for _ in range(self.order + 1):
			xs.append(
				self.benchmark.f(xs[-1])
			)
		z = np.concatenate(xs[:-1])
		zp = np.concatenate(xs[1:])

		self.obj = self.obj + cvxpy.log(
			 z @ self.P @ z
			- zp @ self.P @ zp
		)

class Verifier():
	def __init__(self, benchmark : Benchmark, order=2):
		self.benchmark = benchmark
		self.model = grb_model(convex=False)

		x = grb_cont_var(self.model, (benchmark.dimension,))
		self.xs = [x]
		for _ in range(order + 1):
			self.xs.append(
				self.benchmark.gurobi_formulation(self.model, self.xs[-1])
			)
		self.z = grb_concatenate(self.model, self.xs[:-1])
		self.zp = grb_concatenate(self.model, self.xs[1:])

	def verify(self, P):
		self.model.setObjective(self.zp @ P @ self.zp - self.z @ P @ self.z, GRB.MAXIMIZE)
		self.model.optimize()
		obj = self.model.ObjVal

		if obj > -1e-8:
			return False, self.xs[0].x
		else:
			return True, None

class Cegis():
	def __init__(self, benchmark, order):
		self.learner = Learner(benchmark, order)
		self.verifier = Verifier(benchmark, order)
		self.P : np.array = None

	def solve(self):
		while True:
			print_segline('Learner')
			P = self.learner.learn()

			print_segline("Verifier")
			verified, ctx = self.verifier.verify(P)
			if verified:
				print("verified")
				self.P = P
				return P
			else:
				print("counter example found", ctx)
				self.learner.augment(ctx)



if __name__ == '__main__':
	pwa = PWA()
	cegis = Cegis(pwa, order=0)
	cegis.solve()