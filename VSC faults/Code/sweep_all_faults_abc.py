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


def fVabc_balanced(Iabc):
    Ia = Iabc[0]
    Ib = Iabc[1]
    Ic = Iabc[2]
    
    Va = 1 / (Zf + Z2) * (Ia * (Z1 * Z2 + Z2 * Zf + Zf * Z1) + Vga * Zf)
    Vb = 1 / (Zf + Z2) * (Ib * (Z1 * Z2 + Z2 * Zf + Zf * Z1) + Vgb * Zf)
    Vc = 1 / (Zf + Z2) * (Ic * (Z1 * Z1 + Z2 * Zf + Zf * Z1) + Vgc * Zf)
    Vabc = np.array([Va, Vb, Vc])
    return Vabc


def fVabc_LG(Iabc):
    Ia = Iabc[0]
    Ib = Iabc[1]
    Ic = Iabc[2]

    Va = 1 / (Z2 + Zf) * (Ia * (Z1 * Z2 + Z1 * Zf + Z2 * Zf) + Vga * Zf)
    Vb = Vgb + Ib * (Z1 + Z2)
    Vc = Vgc + Ic * (Z1 + Z2)

    Vabc = np.array([Va, Vb, Vc])
    return Vabc


def fVabc_LL(Iabc):
    Ia = Iabc[0]
    Ib = Iabc[1]
    Ic = Iabc[2]

    Vx = 1 / ((Zf + Z2) * (Zf + 2 * Z2)) * (Ib * (Z2 * Zf * Zf + 2 * Z2 * Z2 * Zf + Z2 * Z2 * Z2) + Ic * (Z2 * Z2 * Zf + Z2 * Z2 * Z2) + Vgb * (Zf * Zf + 2 * Zf * Z2 + Z2 * Z2) + Vgc * (Z2 * Zf + Z2 * Z2))
    Vy = 1 / (Zf + 2 * Z2) * (Ic * (Z2 * (Zf + Z2)) + Vgc * (Zf + Z2) + Ib * Z2 * Z2 + Vgb * Z2)

    Va = Ia * (Z1 + Z2) + Vga
    Vb = Vx + Ib * Z1
    Vc = Vy + Ic * Z1

    Vabc = np.array([Va, Vb, Vc])
    return Vabc


def fVabc_LLG(Iabc):
    Ia = Iabc[0]
    Ib = Iabc[1]
    Ic = Iabc[2]

    Va = Ia * (Z1 + Z2) + Vga
    Vb = Z1 * Ib + (Z2 * Zf * (Ib + Ic) + Zf * (Vgb + Vgc)) / (2 * Zf + Z2)
    Vc = Z1 * Ic + (Z2 * Zf * (Ib + Ic) + Zf * (Vgb + Vgc)) / (2 * Zf + Z2)

    Vabc = np.array([Va, Vb, Vc])
    return Vabc


# initialization
Zf = 0.1 * 1j
Z2 = 0.01 + 0.05 * 1j
Z1 = 0.01 + 0.1 * 1j
Imax = 1

Vga = 1
alph = np.exp(1j * -120 * np.pi / 180)
Vgb = 1 * alph
Vgc = 1 * alph ** 2

Ia_re = 0
Ia_im = 0
Ib_re = 0
Ib_im = 0
Ic_re = 0
Ic_im = 0
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
    Ia_re = -Imax + 2 * Imax * kk / n_iter

    for ll in range(n_iter):
        Ia_im = -Imax + 2 * Imax * ll / n_iter
        if Ia_im ** 2 + Ia_re ** 2 > Imax ** 2:
            Ia_im = np.sign(Ia_im) * np.sqrt(Imax ** 2 - Ia_re ** 2)

        for mm in range(n_iter):
            Ib_re = -Imax + 2 * Imax * mm / n_iter

            for nn in range(n_iter):
                Ib_im = -Imax + 2 * Imax * nn / n_iter
                if Ib_im ** 2 + Ib_re ** 2 > Imax ** 2:
                    Ib_im = np.sign(Ib_im) * np.sqrt(Imax ** 2 - Ib_re ** 2)
                Ic_re = -Ia_re - Ib_re
                Ic_im = -Ia_im - Ib_im
                if not Ic_re ** 2 + Ic_im ** 2 > Imax ** 2:  # Imax ** 2 instead of Imax!
                    Ia = Ia_re + 1j * Ia_im
                    Ib = Ib_re + 1j * Ib_im
                    Ic = Ic_re + 1j * Ic_im
                    Iabc = np.array([Ia, Ib, Ic])
                    Vabc = fVabc_balanced(Iabc)  # change the function accordingly
                    ang_shift = np.angle(Vabc[0])

                    Ia_vec.append(Iabc[0] * np.exp(- 1j * ang_shift))
                    Ib_vec.append(Iabc[1] * np.exp(- 1j * ang_shift))
                    Ic_vec.append(Iabc[2] * np.exp(- 1j * ang_shift))
                    Iabc = np.array([Ia_vec[-1], Ib_vec[-1], Ic_vec[-1]])
                    I012 = Vabc_to_012(Iabc)
                    I0_vec.append(I012[0])
                    I1_vec.append(I012[1])
                    I2_vec.append(I012[2])

                    Va_vec.append(Vabc[0] * np.exp(- 1j * ang_shift))
                    Vb_vec.append(Vabc[1] * np.exp(- 1j * ang_shift))
                    Vc_vec.append(Vabc[2] * np.exp(- 1j * ang_shift))
                    Vabc = np.array([Va_vec[-1], Vb_vec[-1], Vc_vec[-1]])
                    V012 = Vabc_to_012(Vabc)
                    V0_vec.append(V012[0])
                    V1_vec.append(V012[1])
                    V2_vec.append(V012[2])

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


print('Objective function: ', V_obj_min)
print('abc currents: ', Ia_vec[ind_min], Ib_vec[ind_min], Ic_vec[ind_min])
print('012 currents: ', I0_vec[ind_min], I1_vec[ind_min], I2_vec[ind_min])
print('|abc voltages|: ', abs(Va_vec[ind_min]), abs(Vb_vec[ind_min]), abs(Vc_vec[ind_min]))
print('|012 voltages|: ', V0_abs_vec[ind_min], V1_abs_vec[ind_min], V2_abs_vec[ind_min])

# plt.scatter(V1_abs_vec, V2_abs_vec, s=0.5)
# plt.xlabel('|V+|')
# plt.ylabel('|V-|')
# plt.show()


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

ax1.plot(np.real(Ia_vec), V_obj_vec, '.', markersize = 1)
ax1.set_xlabel('Ia_re')
ax1.set_ylabel('f')

ax2.plot(np.imag(Ia_vec), V_obj_vec, '.', markersize = 1)
ax2.set_xlabel('Ia_im')
ax2.set_ylabel('f')

ax3.plot(np.real(Ib_vec), V_obj_vec, '.', markersize = 1)
ax3.set_xlabel('Ib_re')
ax3.set_ylabel('f')

ax4.plot(np.imag(Ib_vec), V_obj_vec, '.', markersize = 1)
ax4.set_xlabel('Ib_im')
ax4.set_ylabel('f')

ax5.plot(np.real(Ic_vec), V_obj_vec, '.', markersize = 1)
ax5.set_xlabel('Ic_re')
ax5.set_ylabel('f')

ax6.plot(np.imag(Ic_vec), V_obj_vec, '.', markersize = 1)
ax6.set_xlabel('Ic_im')
ax6.set_ylabel('f')

plt.show()
