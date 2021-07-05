import numpy as np
from fOpt_1VSC import fOptimal_mystic
from fGCP_1VSC import fGCP_1vsc
from fGCN_1VSC import fGCN_1vsc
from Plots import fPlots
from Functions_main import fZ_rx, fY_fault, x012_to_abc
import pandas as pd
np.set_printoptions(precision=4)
import matplotlib.pyplot as plt

# Data
V_mod = 1
Imax = 1
Zv1 = 0.01 + 0.05 * 1j
Zt = 0.01 + 0.1 * 1j
Zs_i = 6.674e-5 + 1j * 2.597e-4  # series impedances in pu/km
Zp_i = - 1j * 77.372  # parallel impedance in pu.km
Y_con = [0, 0, 0]  # Yab, Ybc, Yac
Y_gnd = [0, 0, 0]  # Yag, Ybg, Ycg
lam_vec = [1, 1]  # V1p, V2p, V1n, V2n
# Ii_t = [0.8178, 0.4075, -0.8483, 0.5295, 0.0305, -0.9371]
# Ii_t = [ 0.824, 0.4774, -0.8523,  0.5216,  0.0281, -0.9989]
# Ii_t = [ 0.8563,  0.5165, -0.8756,  0.4832,  0.0193, -0.9998]
# Ii_t = [ 0.8257,  0.5641, -0.902,   0.4318,  0.0767, -0.996 ]
# Ii_t = [ 0.8244,  0.5656, -0.9022,  0.4314,  0.0775, -0.997 ]
# Ii_t = [ 0.7945,  0.6026, -0.921,   0.3892,  0.1261, -0.992 ]
# Ii_t = [ 0.6811,  0.6874, -0.9621,  0.2716,  0.2814, -0.9591]
# Ii_t = [ 0.6498,  0.6478, -0.9323,  0.2602,  0.282,  -0.9084]
# Ii_t = [ 0.6769,  0.7091, -0.9698,  0.2442,  0.2931, -0.9535]
Ii_t = [ 0.6072,  0.703,  -0.9831,  0.1819,  0.3757, -0.8846]
type_f = 'opt_LL_'
folder = 'Results_1conv_largerZ/'

# RX variation
n_p = 50
# [RX_vec, Zin_vec] = fZ_rx(5, 0.1, n_p, abs(Zv1))  # lim1, lim2, n_p, Zthmod
Yf_vec = fY_fault(1.5, 50, n_p)
# Store data
Vp1_vec = []
Vn1_vec = []

Ip1_re_vec = []
Ip1_im_vec = []

In1_re_vec = []
In1_im_vec = []
f_vec = []
dist_vec = []

# Optimize cases
for iik in range(0, n_p):
    # print(iik)
    # Initialize data
    # Y_con = [0.999 * Yf_vec[iik], Yf_vec[iik], 1.001 * Yf_vec[iik]]  # test fixing
    # Y_con = [Yf_vec[iik], Yf_vec[iik], Yf_vec[iik]]
    # Y_gnd = [Yf_vec[iik], Yf_vec[iik], Yf_vec[iik]]
    # Y_gnd = [Yf_vec[iik], 0, 0]
    Y_con = [Yf_vec[iik], 0, 0]
    # Y_con = [1000, 0, 0]
    # Y_gnd = [Yf_vec[iik], 0, 0]
    # Zv1 = Zin_vec[iik]
    # Zt = Zv1  # I try this


    # Cable
    # iik = iik / 10  # remove
    # dist_vec.append(iik)
    # Zp = Zp_i / iik 
    # Zs = Zs_i * iik
    # Vth_1 = V_mod * Zp * Zp / (2 * Zt * Zp + Zp * Zs + Zp * Zp + Zt * Zs) 
    # Ztt = (Zp * Zp * Zt + Zs * Zp * Zp + Zt * Zs * Zp) / (2 * Zp * Zt + Zp * Zp + Zs * Zp + Zt * Zs)


    # Call optimization
    x_opt = fOptimal_mystic(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec, Ii_t)
    Ii_t = x_opt[4][0]  # uncomment only for OPT
    # x_opt = fGCP_1vsc(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec)
    # x_opt = fGCN_1vsc(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec)
    

    # Cable:
    # x_opt = fOptimal_mystic(Vth_1, Imax, Zv1, Ztt, Y_con, Y_gnd, lam_vec, Ii_t)
    # Ii_t = x_opt[4][0]
    # x_opt = fGCP_1vsc(Vth_1, Imax, Zv1, Ztt, Y_con, Y_gnd, lam_vec)
    # x_opt = fGCN_1vsc(Vth_1, Imax, Zv1, Ztt, Y_con, Y_gnd, lam_vec)


    Vp1_vec.append(x_opt[2][0])
    Vn1_vec.append(x_opt[3][0])

    Ip1_re_vec.append(np.real(x_opt[0][0]))
    Ip1_im_vec.append(np.imag(x_opt[0][0]))
    
    In1_re_vec.append(np.real(x_opt[1][0]))
    In1_im_vec.append(np.imag(x_opt[1][0]))

    I_vsc1 = [0, x_opt[0][0], x_opt[1][0]]
    I_vsc1_abc = x012_to_abc(I_vsc1)

    ff_obj = lam_vec[0] * abs((1 - abs(Vp1_vec[-1]))) + lam_vec[1] * abs((0 + abs(Vn1_vec[-1])))
    f_vec.append(ff_obj)



# Save csv
x_vec = Yf_vec
for ll in range(len(x_vec)):  # to store Zf and not Yf
    x_vec[ll] = 1 / x_vec[ll]

# x_vec = RX_vec

# x_vec = dist_vec

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

# for kk in range(len(x_vec)):
    # print(x_vec[kk])