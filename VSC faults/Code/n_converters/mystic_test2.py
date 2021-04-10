import numpy as np
import mystic as my
N = 12 #1000 # N=len(x)
M = 1e10 # max of c_i
K = 1000 # max of sum(x)
Q = 4 # 40 # npop = N*Q
G = 200 # gtol

# arrays of fixed values
a = np.random.rand(N)
b = 11

# build objective
def cost_factory(b, max=False):
    i = -1 if max else 1
    def cost(x):
        d = 1. / (np.exp(x))
        return b * np.sum(d * x)
    return cost

objective = cost_factory(b, max=True)
bounds = [(0., K)] * N

def penalty_norm(x): # < 0
    return np.sum(x) - K

# build penalty: sum(x) <= K
@my.penalty.linear_inequality(penalty_norm, k=1e12)
def penalty(x):
    return 0.0

# uncomment if want hard constraint of sum(x) == K
@my.constraints.normalized(mass=1000)
def constraints(x):
    return x

result = my.solvers.diffev2(objective, x0=bounds, bounds=bounds, penalty=penalty, constraints=constraints, npop=N*Q, ftol=1e-8, gtol=G, disp=True, full_output=True, cross=.9, scale=.8,)#, map=p.map)