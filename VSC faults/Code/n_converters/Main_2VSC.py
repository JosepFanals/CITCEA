import numpy as np
# from fOptimal_2VSC import fOptimal
from fOpt_2VSC import fOptimal_mystic
from Plots import fPlots
from Functions import fZ_rx, fY_fault, x012_to_abc, f_lam, predictive
import pandas as pd
np.set_printoptions(precision=4)
import matplotlib.pyplot as plt

# Data
V_mod = 1
Imax = 1
Zv1 = 0.01 + 0.05 * 1j
Zv2 = 0.02 + 0.06 * 1j
Zt = 0.01 + 0.1 * 1j
Y_con = [10000, 0, 0]  # Yab, Ybc, Yac
Y_gnd = [100, 0, 0]  # Yag, Ybg, Yc
lam_vec = [1, 1, 1, 1]  # V1p, V2p, V1n, V2n
Ii_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # currents initialization: Ia1re, Ia1im, ...
Ii_t = [0.95, -0.31, -0.91, 0.27, -0.04, 0.37, 0.86, -0.51, 0.01, 0.03, -0.87, 0.48]  # currents initialization: Ia1re, Ia1im, ...
Ii_t0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # currents initialization: Ia1re, Ia1im, ...
type_f = 'opt_LLG_'
folder = 'Results_2conv_RX_v1/'

# RX variation
n_p = 20
[RX_vec, Zin_vec] = fZ_rx(0.5, 1.5, n_p, abs(Zv1))  # lim1, lim2, n_p, Zthmod
# Yf_vec = fY_fault(15, 200, n_p)  # for values big enough to have a severe fault
# lam1_vec = f_lam(1.0, 0.0, n_p)
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
f_vec = []

# Optimize cases
for iik in range(n_p):
    # Initialize data
    # Y_con = [Yf_vec[iik], Yf_vec[iik], Yf_vec[iik]]
    # Y_gnd = [Yf_vec[iik], Yf_vec[iik], Yf_vec[iik]]
    # Y_gnd = [Yf_vec[iik], 0, 0]
    # Y_con = [Yf_vec[iik], 0, 0]
    # Y_con = [1000, 0, 0]
    # Y_gnd = [Yf_vec[iik], 0, 0]
    Zv1 = Zin_vec[iik]
    # lam_vec = [lam1_vec[iik], 1 - lam1_vec[iik], 0, 0]

    if iik > 0:
        Iit1 = Iit2

    # Call optimization
    # x_opt = fOptimal(V_mod, Imax, Zv1, Zv2, Zt, Y_con, Y_gnd, lam_vec, Ii_t)
    x_opt = fOptimal_mystic(V_mod, Imax, Zv1, Zv2, Zt, Y_con, Y_gnd, lam_vec, Ii_t)

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
    # f_vec.append(np.abs(x_opt[8][0]))

    I_vsc1 = [0, x_opt[0][0], x_opt[1][0]]
    I_vsc2 = [0, x_opt[2][0], x_opt[3][0]]
    I_vsc1_abc = x012_to_abc(I_vsc1)
    I_vsc2_abc = x012_to_abc(I_vsc2)

    # ----------------------------

    # ff_obj = np.real(lam_vec[0] * (1 - Vp1_vec[-1] * np.conj(Vp1_vec[-1])) ** 2 + lam_vec[1] * (0 + Vn1_vec[-1] * np.conj(Vn1_vec[-1])) ** 2 + lam_vec[2] * (1 - Vp2_vec[-1] * np.conj(Vp2_vec[-1])) ** 2 + lam_vec[3] * (0 + Vn2_vec[-1] * np.conj(Vn2_vec[-1])) ** 2)
    # ff_obj = np.real(lam_vec[0] * (1 - abs(Vp1_vec[-1])) ** 2 + lam_vec[1] * (0 + abs(Vn1_vec[-1])) ** 2 + lam_vec[2] * (1 - abs(Vp2_vec[-1])) ** 2 + lam_vec[3] * (0 + abs(Vn2_vec[-1])) ** 2)
    # ff_obj = lam_vec[0] * (1 - abs(Vp1_vec[-1])) + lam_vec[1] * (0 + abs(Vn1_vec[-1])) + lam_vec[2] * (1 - abs(Vp2_vec[-1])) + lam_vec[3] * (0 + abs(Vn2_vec[-1]))
    # ff_obj = lam_vec[0] * (1 - abs(Vp1_vec[-1])) ** 2 + lam_vec[1] * (0 + abs(Vn1_vec[-1])) ** 2 
    ff_obj = lam_vec[0] * abs(1 - abs(Vp1_vec[-1])) + lam_vec[1] * abs(0 - abs(Vn1_vec[-1])) + lam_vec[2] * abs(1 - abs(Vp2_vec[-1])) + lam_vec[3] * abs(0 - abs(Vn2_vec[-1]))
    f_vec.append(ff_obj)
    # print(ff_obj)

    # Ii_t = [np.real(I_vsc1_abc[0]), np.imag(I_vsc1_abc[0]), np.real(I_vsc1_abc[1]), np.imag(I_vsc1_abc[1]), np.real(I_vsc1_abc[2]), np.imag(I_vsc1_abc[2]),  np.real(I_vsc2_abc[0]), np.imag(I_vsc2_abc[0]), np.real(I_vsc2_abc[1]), np.imag(I_vsc2_abc[1]), np.real(I_vsc2_abc[2]), np.imag(I_vsc2_abc[2])]
    # Ii_t = x_opt[8][0]
    Iit2 = x_opt[8][0]
    # print(Ii_t)
    print(Iit2)
    if iik > 10:
        # Ii_t = predictive(Iit2, Iit1, 0.01)
        Ii_t = Iit2
    else:
        Ii_t = Iit2



# Save csv
# x_vec = Yf_vec
x_vec = RX_vec
# x_vec = lam1_vec
pcnt = 1
n_pp = int((1-pcnt) * n_p)
fPlots(x_vec, Vp1_vec, folder + type_f + 'Vp1')
fPlots(x_vec, Vp2_vec, folder + type_f + 'Vp2')
fPlots(x_vec, Vn1_vec, folder + type_f + 'Vn1')
fPlots(x_vec, Vn2_vec, folder + type_f + 'Vn2')

fPlots(x_vec, Ip1_re_vec, folder + type_f + 'Ip1re')
fPlots(x_vec, Ip1_im_vec, folder + type_f + 'Ip1im')
fPlots(x_vec, Ip2_re_vec, folder + type_f + 'Ip2re')
fPlots(x_vec, Ip2_im_vec, folder + type_f + 'Ip2im')

fPlots(x_vec, In1_re_vec, folder + type_f + 'In1re')
fPlots(x_vec, In1_im_vec, folder + type_f + 'In1im')
fPlots(x_vec, In2_re_vec, folder + type_f + 'In2re')
fPlots(x_vec, In2_im_vec, folder + type_f + 'In2im')

fPlots(x_vec, f_vec, folder + type_f + 'f_obj')

# Plots
fig, axs = plt.subplots(4, 2)
axs[0, 0].plot(x_vec[n_pp:], Ip1_re_vec[n_pp:])
axs[0, 0].plot(x_vec[n_pp:], Ip2_re_vec[n_pp:])
axs[0, 0].set_title('I+re')
axs[0, 1].plot(x_vec[n_pp:], Ip1_im_vec[n_pp:])
axs[0, 1].plot(x_vec[n_pp:], Ip2_im_vec[n_pp:])
axs[0, 1].set_title('I+im')
axs[1, 0].plot(x_vec[n_pp:], In1_re_vec[n_pp:])
axs[1, 0].plot(x_vec[n_pp:], In2_re_vec[n_pp:])
axs[1, 0].set_title('I-re')
axs[1, 1].plot(x_vec[n_pp:], In1_im_vec[n_pp:])
axs[1, 1].plot(x_vec[n_pp:], In2_im_vec[n_pp:])
axs[1, 1].set_title('I-im')
axs[2, 0].plot(x_vec[n_pp:], Vp1_vec[n_pp:])
axs[2, 0].plot(x_vec[n_pp:], Vp2_vec[n_pp:])
axs[2, 0].set_title('Vp')
axs[2, 1].plot(x_vec[n_pp:], Vn1_vec[n_pp:])
axs[2, 1].plot(x_vec[n_pp:], Vn2_vec[n_pp:])
axs[2, 1].set_title('Vn')
axs[3, 0].plot(x_vec[n_pp:], f_vec[n_pp:])
axs[3, 0].plot(x_vec[n_pp:], f_vec[n_pp:])
axs[3, 0].set_title('f')


# plt.plot(x_vec, Ip1_re_vec)
plt.show()