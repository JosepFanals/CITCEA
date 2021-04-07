import numpy as np
from fOptimal_2VSC import fOptimal
from Plots import fPlots
import pandas as pd
np.set_printoptions(precision=4)

V_mod = 1
Imax = 1
Zv1 = 0.01 + 0.05 * 1j
Zv2 = 0.02 + 0.06 * 1j
Zt = 0.01 + 0.1 * 1j
Y_con = [0, 0, 0]  # Yab, Ybc, Yac
Y_gnd = [15, 0, 0]  # Yag, Ybg, Yc
lam_vec = [1, 1, 1, 1]  # V1p, V2p, V1n, V2n
type_f = 'LG'
folder = 'Data1'

x_opt = fOptimal(V_mod, Imax, Zv1, Zv2, Zt, Y_con, Y_gnd, lam_vec)
print('PCC1 +- absolute voltages: ', x_opt[4], x_opt[5])
print('PCC2 +- absolute voltages: ', x_opt[6], x_opt[7])
print('VSC1 +- injected currents: ', x_opt[0], x_opt[1])
print('VSC2 +- injected currents: ', x_opt[2], x_opt[3])
print('Objective function:        ', x_opt[8])


