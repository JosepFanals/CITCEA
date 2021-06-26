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
Ii_t =  [-0.5015, -0.8652, -0.4985,  0.8669,  1.    , -0.0018]
type_f = 'opt_3x_'
folder = 'Results_1conv/'

# RX variation
n_p = 50
# [RX_vec, Zin_vec] = fZ_rx(5, 0.1, n_p, abs(Zv1))  # lim1, lim2, n_p, Zthmod
Yf_vec = fY_fault(5, 50, n_p)

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
    Y_con = [Yf_vec[iik], Yf_vec[iik], Yf_vec[iik]]
    # Y_gnd = [Yf_vec[iik], Yf_vec[iik], Yf_vec[iik]]
    # Y_gnd = [Yf_vec[iik], 0, 0]
    # Y_con = [Yf_vec[iik], 0, 0]
    # Y_con = [1000, 0, 0]
    # Y_gnd = [Yf_vec[iik], 0, 0]
    # Zv1 = Zin_vec[iik]
    # Zt = Zv1  # I try this


    # Cable
    # dist_vec.append(iik)
    # Zp = Zp_i / iik 
    # Zs = Zs_i * iik
    # Vth_1 = V_mod * Zp * Zp / (2 * Zt * Zp + Zp * Zs + Zp * Zp + Zt * Zs) 
    # Ztt = (Zp * Zp * Zt + Zs * Zp * Zp + Zt * Zs * Zp) / (2 * Zp * Zt + Zp * Zp + Zs * Zp + Zt * Zs)


    # Call optimization
    x_opt = fOptimal_mystic(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec, Ii_t)
    Ii_t = x_opt[4][0]
    # x_opt = fGCP_1vsc(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec)
    # x_opt = fGCN_1vsc(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec)
    

    # Cable:
    # x_opt = fOptimal_mystic(abs(Vth_1), Imax, Zv1, Ztt, Y_con, Y_gnd, lam_vec, Ii_t)
    # x_opt = fGridCode(abs(Vth_1), Imax, Zv1, Ztt, Y_con, Y_gnd, lam_vec, Ii_t)  # adaptative
    # x_opt = fGC_static(abs(Vth_1), Imax, Zv1, Ztt, Y_con, Y_gnd, lam_vec, Ii_t)

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