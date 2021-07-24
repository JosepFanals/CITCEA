import numpy as np
from fOpt_2VSC import fOptimal_mystic
from fGCP_2VSC import fGCP_2vsc
from fGCN_2VSC import fGCN_2vsc
from Plots import fPlots
from Functions_main import fZ_rx, fY_fault, x012_to_abc, f_lam, predictive
import pandas as pd
np.set_printoptions(precision=4)
import matplotlib.pyplot as plt

# Data
V_mod = 1
Imax = 1
Zv1 = 0.01 + 0.05 * 1j
Zv2 = 0.02 + 0.06 * 1j
Zt = 0.01 + 0.1 * 1j
Y_con = [10, 0, 0]  # Yab, Ybc, Yac
Y_gnd = [0, 0, 0]  # Yag, Ybg, Yc
lam_vec = [1, 1, 1, 1]  # V1p, V2p, V1n, V2n
# Ii_t = [ 0.3773, -0.9262, -0.0663,  0.9977, -0.3111, -0.0713,  0.41,   -0.9122, -0.1685, 0.9857, -0.2415, -0.0733]
# Ii_t = [ 0.841,   0.533,  -0.8871,  0.4602,  0.0468, -0.9933,  0.5829,  0.0047, -0.8003, 0.6003,  0.2194, -0.6048]
# Ii_t = [ 0.8458,  0.5332, -0.8884,  0.4591,  0.0426, -0.9926,  0.8899,  0.0047, -0.8004, 0.6001, -0.0889, -0.606 ]
# Ii_t = [ 0.8458,  0.5332, -0.8884,  0.4591,  0.0426, -0.9926,  0.8899,  0.0047, -0.8004, 0.6001, -0.0889, -0.606 ]
# Ii_t = [ 0.7483,  0.6631, -0.9487,  0.3164,  0.201,  -0.9797,  0.86,    0.5101, -0.872, 0.4896,  0.0122, -0.9998]
# Ii_t = [ 0.6717,  0.7408, -0.9775,  0.2111,  0.306,  -0.9521,  0.8568,  0.5156, -0.8751, 0.4842,  0.0184, -0.9999]
Ii_t = [ 0.7017,  0.7125, -0.968,   0.2514,  0.2663, -0.9639,  0.8506,  0.5259, -0.8808,  0.4736,  0.0304, -0.9996]
type_f = 'opt_LL_'
folder = 'Results_2conv/'

# RX variation
n_p = 100
[RX_vec, Zin_vec] = fZ_rx(5, 0.1, n_p, abs(Zv1))  # lim1, lim2, n_p, Zthmod
# Yf_vec = fY_fault(20, 200, n_p)  # for values big enough to have a severe fault
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
    # Y_con = [0, 0, 0]
    # Y_gnd = [Yf_vec[iik], 0, 0]
    Zv1 = Zin_vec[iik]
    # lam_vec = [lam1_vec[iik], 1 - lam1_vec[iik], 0, 0]


    # Call optimization
    x_opt = fOptimal_mystic(V_mod, Imax, Zv1, Zv2, Zt, Y_con, Y_gnd, lam_vec, Ii_t)
    # x_opt = fGCP_2vsc(V_mod, Imax, Zv1, Zv2, Zt, Y_con, Y_gnd, lam_vec)
    # x_opt = fGCN_2vsc(V_mod, Imax, Zv1, Zv2, Zt, Y_con, Y_gnd, lam_vec)

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

    ff_obj = lam_vec[0] * abs(1 - abs(Vp1_vec[-1])) + lam_vec[1] * abs(0 - abs(Vn1_vec[-1])) + lam_vec[2] * abs(1 - abs(Vp2_vec[-1])) + lam_vec[3] * abs(0 - abs(Vn2_vec[-1]))
    f_vec.append(ff_obj)

    # only uncomment for OPT
    Iit2 = x_opt[8][0]
    print(Iit2)
    Ii_t = Iit2
    # print(abs(Vp1_vec[-1]), abs(Vp2_vec[-1]), abs(Vn1_vec[-1]), abs(Vn2_vec[-1]))




# Save csv
# x_vec = Yf_vec
# for ll in range(len(x_vec)):  # to store Zf and not Yf
    # x_vec[ll] = 1 / x_vec[ll]

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