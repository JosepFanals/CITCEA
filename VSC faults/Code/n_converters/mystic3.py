def objective(x):
    return 1

equations_c = """
x0 + x1 + x2 == 0.3
"""

def obj_fun(x):
    return x[0] ** 2 + x[1] - x[2] + 0.56

bounds = [(-1,1),(-1,1),(-1,1)] #unnecessary
xx0 = [0,0,0]

from mystic.symbolic import generate_constraint, generate_solvers, simplify
from mystic.symbolic import generate_penalty, generate_conditions    
from mystic.solvers import diffev2, fmin_powell, fmin

cf = generate_constraint(generate_solvers(simplify(equations_c)))

result = fmin(obj_fun, x0=xx0, bounds=bounds, \
                 constraints=cf, \
                 npop=40, gtol=50, disp=False, full_output=True)

print(result)