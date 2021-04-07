import numpy as np
from fOptimal_2VSC import fOptimal
from Plots import fPlots
from Functions import fZ_rx
import pandas as pd
np.set_printoptions(precision=4)

# Data
V_mod = 1
Imax = 1
Zv1 = 0.01 + 0.05 * 1j
Zv2 = 0.02 + 0.06 * 1j
Zt = 0.01 + 0.1 * 1j
Y_con = [100, 0, 0]  # Yab, Ybc, Yac
Y_gnd = [1, 0, 0]  # Yag, Ybg, Yc
lam_vec = [1, 1, 1, 1]  # V1p, V2p, V1n, V2n
type_f = 'opt_LLG_'
folder = 'Data1/'

# RX variation
n_p = 100
[RX_vec, Zin_vec] = fZ_rx(5, 0.1, n_p, 0.05)  # lim1, lim2, n_p, Zthmod

# Store data
Vp1_vec = []
Vp2_vec = []
Vn1_vec = []
Vn2_vec = []

Ip1_re_vec = []
Ip1_im_vec = []
Ip2_re_vec = []
Ip2_im_vec = []

In1_re_vec = []
In1_im_vec = []
In2_re_vec = []
In2_im_vec = []

# Optimize cases
for iik in range(n_p):
    # Initialize data
    Zv1 = Zin_vec[iik]

    # Call optimization
    x_opt = fOptimal(V_mod, Imax, Zv1, Zv2, Zt, Y_con, Y_gnd, lam_vec)
    Vp1_vec.append(x_opt[4][0])
    Vp2_vec.append(x_opt[6][0])
    Vn1_vec.append(x_opt[5][0])
    Vn2_vec.append(x_opt[7][0])

    Ip1_re_vec.append(np.real(x_opt[0][0]))
    Ip1_im_vec.append(np.imag(x_opt[0][0]))
    Ip2_re_vec.append(np.real(x_opt[2][0]))
    Ip2_im_vec.append(np.imag(x_opt[2][0]))
    
    In1_re_vec.append(np.real(x_opt[1][0]))
    In1_im_vec.append(np.imag(x_opt[1][0]))
    In2_re_vec.append(np.real(x_opt[3][0]))
    In2_im_vec.append(np.imag(x_opt[3][0]))

# Save csv
fPlots(RX_vec, Vp1_vec, folder + type_f + 'Vp1')
fPlots(RX_vec, Vp2_vec, folder + type_f + 'Vp2')
fPlots(RX_vec, Vn1_vec, folder + type_f + 'Vn1')
fPlots(RX_vec, Vn2_vec, folder + type_f + 'Vn2')

fPlots(RX_vec, Ip1_re_vec, folder + type_f + 'Ip1re')
fPlots(RX_vec, Ip1_im_vec, folder + type_f + 'Ip1im')
fPlots(RX_vec, Ip2_re_vec, folder + type_f + 'Ip2re')
fPlots(RX_vec, Ip2_im_vec, folder + type_f + 'Ip2im')

fPlots(RX_vec, In1_re_vec, folder + type_f + 'In1re')
fPlots(RX_vec, In1_im_vec, folder + type_f + 'In1im')
fPlots(RX_vec, In2_re_vec, folder + type_f + 'In2re')
fPlots(RX_vec, In2_im_vec, folder + type_f + 'In2im')
