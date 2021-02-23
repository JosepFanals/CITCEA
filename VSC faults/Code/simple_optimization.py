import numpy as np
from scipy.optimize import minimize


def objective(x):
    return 1 * (x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])


def constraint1(x):
    return x[0] * x[1] * x[2] * x[3] - 25.0


def constraint2(x):
    sum_sq = 40
    sum_sq += -x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - x[3] ** 2
    return sum_sq


x0 = [1,5,5,1]
print(objective(x0))

b = (1.0,5.0)
bnds = (b,b,b,b)
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'eq', 'fun': constraint2}
cons = [con1, con2]

sol = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
print(sol)