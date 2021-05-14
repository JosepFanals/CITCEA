import numpy as np
# from fOptimal_2VSC import fOptimal
from fOpt_1VSC_new import fOptimal_mystic
from fRopt_1VSC_new import fROptimal_mystic
from fGridCode_1VSC import fGridCode
from Plots import fPlots
from Functions import fZ_rx, fY_fault, x012_to_abc
import pandas as pd
np.set_printoptions(precision=4)
import matplotlib.pyplot as plt

# Data
V_mod = 1
Imax = 1
Zv1 = 0.01 + 0.05 * 1j
Zt = 0.01 + 0.1 * 1j
Y_con = [10, 10, 10]  # Yab, Ybc, Yac
Y_gnd = [0, 0, 0]  # Yag, Ybg, Ycg
# lam_vec = [1, 1]  # V1p, V2p, V1n, V2n
lam_vec = [1, 1]  # V1p, V2p, V1n, V2n
# Ii_t = [0.0168, -0.6161, -0.7091, 0.7052, 0.6924, -0.089]
Ii_t = [0.00, -1, -0.866,  0.500,  0.866,  0.5] 
type_f = 'opt_3x_'
folder = 'Results_1conv_RX_v1/'

# RX variation
n_p = 500
[RX_vec, Zin_vec] = fZ_rx(5, 0.1, n_p, abs(Zv1))  # lim1, lim2, n_p, Zthmod
# Yf_vec = fY_fault(20, 70, n_p)

# Store data
Vp1_vec = []
Vn1_vec = []

Ip1_re_vec = []
Ip1_im_vec = []

In1_re_vec = []
In1_im_vec = []
f_vec = []

# Optimize cases
for iik in range(n_p):
    print(iik)
    # Initialize data
    # Y_con = [Yf_vec[iik], Yf_vec[iik], Yf_vec[iik]]
    # Y_gnd = [Yf_vec[iik], Yf_vec[iik], Yf_vec[iik]]
    # Y_gnd = [Yf_vec[iik], 0, 0]
    # Y_con = [Yf_vec[iik], 0, 0]
    # Y_con = [1000, 0, 0]
    # Y_gnd = [Yf_vec[iik], 0, 0]
    Zv1 = Zin_vec[iik]
    # Zt = Zv1  # I try this

    # Call optimization
    # x_opt = fOptimal(V_mod, Imax, Zv1, Zv2, Zt, Y_con, Y_gnd, lam_vec, Ii_t)
    x_opt = fOptimal_mystic(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec, Ii_t)
    # x_opt = fROptimal_mystic(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec, Ii_t)
    # x_opt = fGridCode(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec, Ii_t)

    Vp1_vec.append(x_opt[2][0])
    Vn1_vec.append(x_opt[3][0])

    Ip1_re_vec.append(np.real(x_opt[0][0]))
    Ip1_im_vec.append(np.imag(x_opt[0][0]))
    
    In1_re_vec.append(np.real(x_opt[1][0]))
    In1_im_vec.append(np.imag(x_opt[1][0]))
    # f_vec.append(np.abs(x_opt[8][0]))

    I_vsc1 = [0, x_opt[0][0], x_opt[1][0]]
    I_vsc1_abc = x012_to_abc(I_vsc1)

    # ----------------------------

    # ff_obj = np.real(lam_vec[0] * (1 - Vp1_vec[-1] * np.conj(Vp1_vec[-1])) ** 2 + lam_vec[1] * (0 + Vn1_vec[-1] * np.conj(Vn1_vec[-1])) ** 2)
    ff_obj = lam_vec[0] * abs((1 - abs(Vp1_vec[-1]))) + lam_vec[1] * abs((0 + abs(Vn1_vec[-1])))
    f_vec.append(ff_obj)

    Ii_t = x_opt[4][0]

    print(ff_obj)
    print(Ii_t)

# Save csv
# x_vec = Yf_vec
# for ll in range(len(x_vec)):  # to store Zf and not Yf
#     x_vec[ll] = 1 / x_vec[ll]

x_vec = RX_vec

pcnt = 1
n_pp = int((1-pcnt) * n_p)
fPlots(x_vec, Vp1_vec, folder + type_f + 'Vp1')
fPlots(x_vec, Vn1_vec, folder + type_f + 'Vn1')

fPlots(x_vec, Ip1_re_vec, folder + type_f + 'Ip1re')
fPlots(x_vec, Ip1_im_vec, folder + type_f + 'Ip1im')

fPlots(x_vec, In1_re_vec, folder + type_f + 'In1re')
fPlots(x_vec, In1_im_vec, folder + type_f + 'In1im')

fPlots(x_vec, f_vec, folder + type_f + 'f_obj')

# Plots
fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(x_vec[n_pp:], Ip1_re_vec[n_pp:])
axs[0, 0].set_title('Axis [0, 0]')
axs[0, 1].plot(x_vec[n_pp:], Ip1_im_vec[n_pp:])
axs[0, 1].set_title('Axis [0, 1]')
axs[1, 0].plot(x_vec[n_pp:], In1_re_vec[n_pp:])
axs[1, 0].set_title('Axis [1, 0]')
axs[1, 1].plot(x_vec[n_pp:], In1_im_vec[n_pp:])
axs[1, 1].set_title('Axis [1, 1]')
axs[2, 0].plot(x_vec[n_pp:], f_vec[n_pp:])
axs[2, 0].set_title('f')
axs[2, 1].plot(x_vec[n_pp:], Vp1_vec[n_pp:])
axs[2, 0].set_title('f')


# plt.plot(x_vec, Ip1_re_vec)
plt.show()