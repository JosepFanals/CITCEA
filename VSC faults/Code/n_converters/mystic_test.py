import numpy as np
import mystic as my
N = 10 #1000 # N=len(x)
M = 1e10 # max of c_i
K = 1000 # max of sum(x)
Q = 4 # 40 # npop = N*Q
G = 200 # gtol

a = 2
b = 3
c = 8


# build objective
def cost_factory(a, b, c):
    return x**2 - x + 2 + a + b + c

objective = cost_factory(a, b, c)
bounds = (0., K)

def penalty_norm(x): # < 0
    return x - K

# build penalty: sum(x) <= K
@my.penalty.linear_inequality(penalty_norm, k=1e12)
def penalty(x):
    return 0.0

# uncomment if want hard constraint of sum(x) == K
@my.constraints.normalized(mass=1000)
def constraints(x):
    return x

result = my.solvers.diffev2(objective, x0=bounds, bounds=bounds, penalty=penalty, constraints=constraints, npop=N*Q, ftol=1e-8, gtol=G, disp=True, full_output=True, cross=.9, scale=.8,)#, map=p.map)