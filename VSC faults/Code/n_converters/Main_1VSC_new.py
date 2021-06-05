import numpy as np
# from fOptimal_2VSC import fOptimal
from fOpt_1VSC_new import fOptimal_mystic
from fRopt_1VSC_new import fROptimal_mystic
from fRopt_1VSC_new2 import fROptimal2_mystic
from fADA2 import fADA2
from fGridCode_1VSC import fGridCode
from fGC_static import fGC_static
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
Zs_i = 6.674e-5 + 1j * 2.597e-4  # series impedances in pu/km
Zp_i = - 1j * 77.372  # parallel impedance in pu.km
Y_con = [10, 0, 0]  # Yab, Ybc, Yac
Y_gnd = [0, 0, 0]  # Yag, Ybg, Ycg
# lam_vec = [1, 1]  # V1p, V2p, V1n, V2n
lam_vec = [1, 1]  # V1p, V2p, V1n, V2n
# Ii_t = [1,  0.00, -1, 0.0 , 0.0,-0.0]
# Ii_t = [1, 1, 1, 1, 1, 1]
# Ii_t = [-0.2012,  0.2415,  0.0137 , 0.0613 , 0.1873 ,-0.3029]
# Ii_t = [-0.4331, -0.8344, -0.563,   0.8265,  0.9962,  0.0076]
# Ii_t = [-0.2003, -0.3512, -0.2041,  0.3493,  0.4044,  0.0024]
# Ii_t = [-0.2166, -0.689, -0.7213, 0.6928, 0.9376, -0.0038]
# Ii_t = [0.7899, -0.4999, -0.862,   0.5071,  0.0719, -0.0071]
# Ii_t = [0.8543, -0.4144, -0.9054, 0.4247, 0.0511, -0.0103]
# Ii_t = [ 0.8656, -0.3836, -0.9168,  0.3995,  0.0513, -0.016]
# Ii_t = [ 0.8791, -0.3682, -0.9218,  0.3877,  0.0429, -0.0196]
Ii_t = [ 0.8894, -0.3522, -0.9263,  0.3771,  0.0371, -0.025 ]
# Ii_t = [0, 0, 0, 0, 0, 0]
type_f = 'sta_LL_'
folder = 'Results_1conv_Zf_sat_v1/'

# RX variation
n_p = 10
# [RX_vec, Zin_vec] = fZ_rx(5, 0.1, n_p, abs(Zv1))  # lim1, lim2, n_p, Zthmod
# Yf_vec = fY_fault(5, 50, n_p)

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
for iik in range(1, n_p):
    # print(iik)
    # Initialize data
    # Y_con = [Yf_vec[iik], Yf_vec[iik], Yf_vec[iik]]
    # Y_gnd = [Yf_vec[iik], Yf_vec[iik], Yf_vec[iik]]
    # Y_gnd = [Yf_vec[iik], 0, 0]
    # Y_con = [Yf_vec[iik], 0, 0]
    # Y_con = [1000, 0, 0]
    # Y_gnd = [Yf_vec[iik], 0, 0]
    # Zv1 = Zin_vec[iik]
    # Zt = Zv1  # I try this

    # Cable
    dist_vec.append(iik)
    Zp = Zp_i / iik 
    Zs = Zs_i * iik

    Vth_1 = V_mod * Zp * Zp / (2 * Zt * Zp + Zp * Zs + Zp * Zp + Zt * Zs) 
    Ztt = (Zp * Zp * Zt + Zs * Zp * Zp + Zt * Zs * Zp) / (2 * Zp * Zt + Zp * Zp + Zs * Zp + Zt * Zs)



    # Call optimization
    # x_opt = fOptimal(V_mod, Imax, Zv1, Zv2, Zt, Y_con, Y_gnd, lam_vec, Ii_t)
    # x_opt = fOptimal_mystic(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec, Ii_t)
    # x_opt = fROptimal_mystic(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec, Ii_t)
    # x_opt = fROptimal2_mystic(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec, Ii_t)
    # x_opt = fGridCode(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec, Ii_t)  # adaptative
    # x_opt = fADA2(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec, Ii_t)
    # x_opt = fGC_static(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec, Ii_t)

    # Cable:
    # x_opt = fOptimal_mystic(abs(Vth_1), Imax, Zv1, Ztt, Y_con, Y_gnd, lam_vec, Ii_t)
    x_opt = fGridCode(abs(Vth_1), Imax, Zv1, Ztt, Y_con, Y_gnd, lam_vec, Ii_t)  # adaptative
    # x_opt = fGC_static(abs(Vth_1), Imax, Zv1, Ztt, Y_con, Y_gnd, lam_vec, Ii_t)

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
    # print(abs(max(abs(Ii_t))))
    # print(max(np.sqrt(Ii_t[0] ** 2 + Ii_t[1] ** 2), np.sqrt(Ii_t[2] ** 2 + Ii_t[3] ** 2), np.sqrt(Ii_t[4] ** 2 + Ii_t[5] ** 2)))

    # print(ff_obj)
    # print(Ii_t)
    # print(Ii_t[0]**2 + Ii_t[1]**2)

    # print(x_opt)

# Save csv
# x_vec = Yf_vec
# for ll in range(len(x_vec)):  # to store Zf and not Yf
    # x_vec[ll] = 1 / x_vec[ll]

# x_vec = RX_vec

x_vec = dist_vec

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