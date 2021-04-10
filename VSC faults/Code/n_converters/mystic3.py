def objective(x):
    return 1

equations = """
x0 + x1 == 0
"""

bounds = [(-1,1),(-1,1)] #unnecessary

from mystic.symbolic import generate_constraint, generate_solvers, simplify
from mystic.symbolic import generate_penalty, generate_conditions    
from mystic.solvers import diffev2

cf = generate_constraint(generate_solvers(simplify(equations)))

result = diffev2(objective, x0=bounds, bounds=bounds, \
                 constraints=cf, \
                 npop=40, gtol=50, disp=False, full_output=True)

print(result)