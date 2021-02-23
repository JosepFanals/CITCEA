import numpy as np
import matplotlib.pyplot as plt


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


def fVabc(Iabc):  # this is for the balanced
    Ia = Iabc[0]
    Ib = Iabc[1]
    Ic = Iabc[2]
    
    Va = 1 / (Zf + Z2) * (Ia * (Z1 * Z2 + Z2 * Zf + Zf * Z1) + Vga * Zf)
    Vb = 1 / (Zf + Z2) * (Ib * (Z1 * Z2 + Z2 * Zf + Zf * Z1) + Vgb * Zf)
    Vc = 1 / (Zf + Z2) * (Ic * (Z1 * Z1 + Z2 * Zf + Zf * Z1) + Vgc * Zf)
    Vabc = np.array([Va, Vb, Vc])
    return Vabc


# initialization
Zf = 0.05 * 1j
Z2 = 0.01 + 0.05 * 1j
Z1 = 0.02 + 0.1 * 1j
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

Ia_re_vec = []
Ib_re_vec = []
Ic_re_vec = []

Ia_im_vec = []
Ib_im_vec = []
Ic_im_vec = []

Va_re_vec = []
Vb_re_vec = []
Vc_re_vec = []

Va_im_vec = []
Vb_im_vec = []
Vc_im_vec = []

V0_re_vec = []
V1_re_vec = []
V2_re_vec = []

V0_im_vec = []
V1_im_vec = []
V2_im_vec = []

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
                if not Ic_re ** 2 + Ic_im ** 2 > Imax:
                    Ia = Ia_re + 1j * Ia_im
                    Ib = Ib_re + 1j * Ib_im
                    Ic = Ic_re + 1j * Ic_im
                    Iabc = np.array([Ia, Ib, Ic])

                    Vabc = fVabc(Iabc)
                    V012 = Vabc_to_012(Vabc)

                    V0_re_vec.append(np.real(V012[0]))
                    V1_re_vec.append(np.real(V012[1]))
                    V2_re_vec.append(np.real(V012[2]))

                    V0_im_vec.append(np.imag(V012[0]))
                    V1_im_vec.append(np.imag(V012[1]))
                    V2_im_vec.append(np.imag(V012[2]))

                    Va_re_vec.append(np.real(Vabc[0]))
                    Vb_re_vec.append(np.real(Vabc[1]))
                    Vc_re_vec.append(np.real(Vabc[2]))

                    Va_im_vec.append(np.imag(Vabc[0]))
                    Vb_im_vec.append(np.imag(Vabc[1]))
                    Vc_im_vec.append(np.imag(Vabc[2]))

                    Ia_re_vec.append(np.real(Iabc[0]))
                    Ib_re_vec.append(np.real(Iabc[1]))
                    Ic_re_vec.append(np.real(Iabc[2]))

                    Ia_im_vec.append(np.imag(Iabc[0]))
                    Ib_im_vec.append(np.imag(Iabc[1]))
                    Ic_im_vec.append(np.imag(Iabc[2]))

                    compt += 1

V1_abs_vec = []
V2_abs_vec = []
V12_abs_vec = []  # maximum difference between abs(V1) and abs(V2)
V12_max = 0
ind_max = 0

for kk in range(compt):
    V1_abs_vec.append(np.sqrt(V1_re_vec[kk] ** 2 + V1_im_vec[kk] ** 2))
    V2_abs_vec.append(np.sqrt(V2_re_vec[kk] ** 2 + V2_im_vec[kk] ** 2))
    V12_diff = V1_abs_vec[kk] - V2_abs_vec[kk]
    V12_abs_vec.append(V12_diff)
    V12_max = max(V12_diff, V12_max)
    if V12_max == V12_diff:
        ind_max = kk

print('Objective function: ', V12_max)
Ia_ff = Ia_re_vec[ind_max] + 1j * Ia_im_vec[ind_max]
Ib_ff = Ib_re_vec[ind_max] + 1j * Ib_im_vec[ind_max]
Ic_ff = Ic_re_vec[ind_max] + 1j * Ic_im_vec[ind_max]
print('abc currents: ', Ia_ff, Ib_ff, Ic_ff)



plt.scatter(V1_abs_vec, V2_abs_vec, s=1)
plt.show()

