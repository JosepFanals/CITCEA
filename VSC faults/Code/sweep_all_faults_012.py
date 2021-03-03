import numpy as np
import matplotlib.pyplot as plt
import sys
from math import *


def progress_bar(i, n, size):
    percent = float(i) / float(n)
    sys.stdout.write("\r" + str(int(i)).rjust(3,'0') + "/" + str(int(n)).rjust(3, '0') + ' [' + '='*ceil(percent*size) + ' '*floor((1-percent)*size) + ']')
    

def Vabc_to_012(Vabc):
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

    V012 = np.dot(T, Vabc)
    return V012


def V012_to_abc(V012):
    T = np.zeros((3,3), dtype=complex)
    T[0,0] = 1
    T[0,1] = 1
    T[0,2] = 1

    T[1,0] = 1
    T[1,1] = 1 * np.exp(- 1j * 2 * np.pi / 3)
    T[1,2] = 1 * np.exp(1j * 2 * np.pi / 3)

    T[2,0] = 1
    T[2,1] = 1 * np.exp(1j * 2 * np.pi / 3)
    T[2,2] = 1 * np.exp(- 1j * 2 * np.pi / 3)

    # Tinv = np.linalg.inv(T)
    Vabc = np.dot(T, V012)
    return Vabc


def fV012_balanced(I012):
    I0 = I012[0]
    I1 = I012[1]
    I2 = I012[2]

    V0 = 0
    V1 = 1 / (Zf + Z2) * (Vg1 * Zf + I1 * (Z1 * Zf + Z1 * Z2 + Zf * Z2))
    V2 = 1 / (Zf + Z2) * (I2 * (Z2 * Zf + Z1 * Z2 + Z1 * Zf))

    V012 = np.array([V0, V1, V2])
    return V012


def fV012_LG(I012):
    I0 = I012[0]
    I1 = I012[1]
    I2 = I012[2]
    
    V0 = - Z2 / (3 * Zf + 3 * Z2) * (1 + I1 * Z2 + I2 * Z2)
    V1 = I1 * (Z1 + Z2) + Vg1 - Z2 / (3 * Zf + 3 * Z2) * (I2 * Z2 + I1 * Z2 + Vg1)
    V2 = I2 * (Z1 + Z2) - Z2 / (3 * Zf + 3 * Z2) * (I2 * Z2 + I1 * Z2 + Vg1)

    V012 = np.array([V0, V1, V2])
    return V012


def fV012_LL(I012):
    I0 = I012[0]
    I1 = I012[1]
    I2 = I012[2]

    V0 = 0
    V1 = Vg1 + I1 * Z1 + I1 * Z2 - Z2 / (2 * Z2 + Zf) * (Vg1 + I1 * Z2 - I2 * Z2)
    V2 = Vg1 + I1 * Z2 + I2 * Z1 - (Z2 + Zf) / (2 * Z2 + Zf) * (Vg1 + I1 * Z2 - I2 * Z2)

    V012 = np.array([V0, V1, V2])
    return V012


def fV012_LLG(I012):
    I0 = I012[0]
    I1 = I012[1]
    I2 = I012[2]

    V1 = I1 * Z1 + (Z2 + 3 * Zf) / (3 * Z2 + 6 * Zf) * (I1 * Z2 + I2 * Z2 + Vg1)
    V2 = I2 * Z1 + (Z2 + 3 * Zf) / (3 * Z2 + 6 * Zf) * (I1 * Z2 + I2 * Z2 + Vg1)
    V0 = Z2 / (3 * Z2 + 6 * Zf) * (I1 * Z2 + I2 * Z2 + Vg1)

    V012 = np.array([V0, V1, V2])
    return V012


# initialization
Zf = 0.1 * 1j
Z2 = 0.01 + 0.05 * 1j
Z1 = 0.01 + 0.1 * 1j
Imax = 1

Vga = 1
alph = np.exp(1j * -120 * np.pi / 180)
Vgb = 1 * alph
Vgc = 1 * alph ** 2
Vg1 = 1  # positive sequence voltage at the grid

I0_re = 0
I0_im = 0
I1_re = 0
I1_im = 0
I2_re = 0
I2_im = 0

n_iter = 40
compt = 0

Ia_vec = []
Ib_vec = []
Ic_vec = []

I0_vec = []
I1_vec = []
I2_vec = []

Va_vec = []
Vb_vec = []
Vc_vec = []

V0_vec = []
V1_vec = []
V2_vec = []

percent = 0
n_points = n_iter ** 1  # only 1st loop
n_compt = 0

for kk in range(n_iter):
    I1_re = -Imax + 2 * Imax * kk / n_iter
    for ll in range(n_iter):
        I1_im = -Imax + 2 * Imax * ll / n_iter
        for mm in range(n_iter):
            I2_re = -Imax + 2 * Imax * mm / n_iter
            for nn in range(n_iter):
                I2_im = -Imax + 2 * Imax * nn / n_iter
                I0_re = 0
                I0_im = 0

                I0 = I0_re + 1j * I0_im
                I1 = I1_re + 1j * I1_im
                I2 = I2_re + 1j * I2_im
                I012 = np.array([I0, I1, I2])

                Iabc = V012_to_abc(I012)
                Ia = Iabc[0]
                Ib = Iabc[1]
                Ic = Iabc[2]
                if not abs(Ia) > Imax and not abs(Ib) > Imax and not abs(Ic) > Imax:   
                    
                    V012 = fV012_LG(I012)  # change the function accordingly
                    Vabc = V012_to_abc(V012)
                    ang_shift = np.angle(Vabc[0])
                    Va_vec.append(Vabc[0] * np.exp(- 1j * ang_shift))
                    Vb_vec.append(Vabc[1] * np.exp(- 1j * ang_shift))
                    Vc_vec.append(Vabc[2] * np.exp(- 1j * ang_shift))
                    Vabc = np.array([Va_vec[-1], Vb_vec[-1], Vc_vec[-1]])
                    V012 = Vabc_to_012(Vabc)
                    V0_vec.append(V012[0])
                    V1_vec.append(V012[1])
                    V2_vec.append(V012[2])

                    Ia_vec.append(Iabc[0] * np.exp(- 1j * ang_shift))
                    Ib_vec.append(Iabc[1] * np.exp(- 1j * ang_shift))
                    Ic_vec.append(Iabc[2] * np.exp(- 1j * ang_shift))
                    Iabc = np.array([Ia_vec[-1], Ib_vec[-1], Ic_vec[-1]])
                    I012 = Vabc_to_012(Iabc)
                    I0_vec.append(I012[0])
                    I1_vec.append(I012[1])
                    I2_vec.append(I012[2])

                    compt += 1

    progress_bar(kk, n_iter, 50)

V0_abs_vec = []
V1_abs_vec = []
V2_abs_vec = []
V12_abs_vec = []  # maximum difference between abs(V1) and abs(V2)
V12_max = 0
V_obj_vec = []
V_obj_min = 1
ind_max = 0
ind_min = 0

for kk in range(compt):
    V0_abs_vec.append(np.abs(V0_vec[kk]))
    V1_abs_vec.append(np.abs(V1_vec[kk]))
    V2_abs_vec.append(np.abs(V2_vec[kk]))

    # objective function:
    V_object = 1 * abs(V1_abs_vec[kk] - 1) + 1 * abs(V2_abs_vec[kk] - 0)
    V_obj_vec.append(V_object)
    V_obj_min = min(V_object, V_obj_min)
    if V_obj_min == V_object:
        ind_min = kk

print('\n')
print('Objective function: ', V_obj_min)
print('abc currents: ', Ia_vec[ind_min], Ib_vec[ind_min], Ic_vec[ind_min])
print('012 currents: ', I0_vec[ind_min], I1_vec[ind_min], I2_vec[ind_min])
print('|abc voltages|: ', abs(Va_vec[ind_min]), abs(Vb_vec[ind_min]), abs(Vc_vec[ind_min]))
print('|012 voltages|: ', V0_abs_vec[ind_min], V1_abs_vec[ind_min], V2_abs_vec[ind_min])


# plt.scatter(V_obj_vec, V2_abs_vec, s=1)
# plt.xlabel('|V+|')
# plt.ylabel('|V-|')
# plt.show()


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

ax1.plot(np.real(I1_vec), V_obj_vec, '.', markersize = 1)
ax1.set_xlabel('I1_re')
ax1.set_ylabel('f')

ax5.plot(np.real(I1_vec[ind_min]), V_obj_vec[ind_min])

ax2.plot(np.imag(I1_vec), V_obj_vec, '.', markersize = 1)
ax2.set_xlabel('I1_im')
ax2.set_ylabel('f')

ax3.plot(np.real(I2_vec), V_obj_vec, '.', markersize = 1)
ax3.set_xlabel('I2_re')
ax3.set_ylabel('f')

ax4.plot(np.imag(I2_vec), V_obj_vec, '.', markersize = 1)
ax4.set_xlabel('I2_im')
ax4.set_ylabel('f')

plt.show()