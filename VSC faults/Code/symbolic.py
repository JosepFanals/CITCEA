import sympy as sym
import numpy as np

sym.init_printing()

z2, zf = sym.symbols('z2 zf')
matrix = sym.Matrix([[-1/z2, -1/z2, 2/z2], 
                     [1/zf + 1/z2, -1/zf, -1/z2],
                     [-1/zf, 1/zf + 1/z2, -1/z2]])

print(matrix.inv())