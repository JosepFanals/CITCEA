import numpy as np
# from fOptimal_2VSC import fOptimal
from fOpt_1VSC import fOptimal_mystic
from Plots import fPlots
from Functions import fZ_rx, fY_fault, x012_to_abc, build_static_objects1
import pandas as pd
np.set_printoptions(precision=4)
import matplotlib.pyplot as plt

V_mod = 1
Imax = 1
Zv1 = 0.01 + 0.05 * 1j
Zt = 0.01 + 0.1 * 1j
Y_con = [0, 0, 0]  # Yab, Ybc, Yac
Y_gnd = [2, 0, 0]  # Yag, Ybg, Ycg

objec_i = build_static_objects1(V_mod, Zv1, Zt, Y_con, Y_gnd)
print(objec_i)

y_mat = objec_i[0]
print(y_mat)
z_mat = np.linalg.inv(y_mat)
print(z_mat)

z_mat3 = z_mat[0:3, 0:6]
print(z_mat3)

T = np.zeros((3,3), dtype=complex)
T[0,0] = 1 / 3
T[0,1] = 1 / 3
T[0,2] = 1 / 3

T[1,0] = 1 / 3
T[1,1] = 1 / 3 * np.exp(1j * 2 * np.pi / 3)
T[1,2] = 1 / 3 * np.exp(- 1j * 2 * np.pi / 3)

T[2,0] = 1 / 3
T[2,1] = 1 / 3 * np.exp(- 1j * 2 * np.pi / 3)
T[2,2] = 1 / 3 * np.exp(1j * 2 * np.pi / 3)

Mm = np.dot(T, z_mat3)
print(Mm)