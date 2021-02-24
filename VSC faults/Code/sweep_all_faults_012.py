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
Z2 = 0.00 + 0.05 * 1j
Z1 = 0.00 + 0.1 * 1j
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

Ia_re_vec = []
Ib_re_vec = []
Ic_re_vec = []

Ia_im_vec = []
Ib_im_vec = []
Ic_im_vec = []

I0_re_vec = []
I1_re_vec = []
I2_re_vec = []

I0_im_vec = []
I1_im_vec = []
I2_im_vec = []

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
                    
                    V012 = fV012_balanced(I012)
                    Vabc = V012_to_abc(V012)

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

                    I0_re_vec.append(np.real(I012[0]))
                    I1_re_vec.append(np.real(I012[1]))
                    I2_re_vec.append(np.real(I012[2]))

                    I0_im_vec.append(np.imag(I012[0]))
                    I1_im_vec.append(np.imag(I012[1]))
                    I2_im_vec.append(np.imag(I012[2]))

                    compt += 1

    n_compt += 1
    percent = n_compt / n_points * 100
    print(str(round(percent, 2)))

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
    V0_abs_vec.append(np.sqrt(V0_re_vec[kk] ** 2 + V0_im_vec[kk] ** 2))
    V1_abs_vec.append(np.sqrt(V1_re_vec[kk] ** 2 + V1_im_vec[kk] ** 2))
    V2_abs_vec.append(np.sqrt(V2_re_vec[kk] ** 2 + V2_im_vec[kk] ** 2))

    # objective: maximize the difference between V+ and V-. Called objective max
    V12_diff = V1_abs_vec[kk] - V2_abs_vec[kk]
    V12_abs_vec.append(V12_diff)
    V12_max = max(V12_diff, V12_max)
    if V12_max == V12_diff:
        ind_max = kk

    # objective: get |V+| close to 1 and |V-| close to 0. Called objective min
    V_object = 0 * abs(V2_abs_vec[kk] - 1) + abs(V2_abs_vec[kk] - 0)
    V_obj_vec.append(V_object)
    V_obj_min = min(V_object, V_obj_min)
    if V_obj_min == V_object:
        ind_min = kk


print('Objective max function: ', V12_max)
Ia_ff = Ia_re_vec[ind_max] + 1j * Ia_im_vec[ind_max]
Ib_ff = Ib_re_vec[ind_max] + 1j * Ib_im_vec[ind_max]
Ic_ff = Ic_re_vec[ind_max] + 1j * Ic_im_vec[ind_max]
I0_ff = I0_re_vec[ind_max] + 1j * I0_im_vec[ind_max]
I1_ff = I1_re_vec[ind_max] + 1j * I1_im_vec[ind_max]
I2_ff = I2_re_vec[ind_max] + 1j * I2_im_vec[ind_max]
Va_ff = Va_re_vec[ind_max] + 1j * Va_im_vec[ind_max]
Vb_ff = Vb_re_vec[ind_max] + 1j * Vb_im_vec[ind_max]
Vc_ff = Vc_re_vec[ind_max] + 1j * Vc_im_vec[ind_max]
print('abc currents: ', Ia_ff, Ib_ff, Ic_ff)
print('012 currents: ', I0_ff, I1_ff, I2_ff)
print('abc voltages: ', Va_ff, Vb_ff, Vc_ff)
print('012 voltages: ', V0_abs_vec[ind_max], V1_abs_vec[ind_max], V2_abs_vec[ind_max])

print('Objective min function: ', V_obj_min)
Ia_ff = Ia_re_vec[ind_min] + 1j * Ia_im_vec[ind_min]
Ib_ff = Ib_re_vec[ind_min] + 1j * Ib_im_vec[ind_min]
Ic_ff = Ic_re_vec[ind_min] + 1j * Ic_im_vec[ind_min]
I0_ff = I0_re_vec[ind_max] + 1j * I0_im_vec[ind_max]
I1_ff = I1_re_vec[ind_max] + 1j * I1_im_vec[ind_max]
I2_ff = I2_re_vec[ind_max] + 1j * I2_im_vec[ind_max]
Va_ff = Va_re_vec[ind_min] + 1j * Va_im_vec[ind_min]
Vb_ff = Vb_re_vec[ind_min] + 1j * Vb_im_vec[ind_min]
Vc_ff = Vc_re_vec[ind_min] + 1j * Vc_im_vec[ind_min]
print('abc currents: ', Ia_ff, Ib_ff, Ic_ff)
print('012 currents: ', I0_ff, I1_ff, I2_ff)
print('abc voltages: ', Va_ff, Vb_ff, Vc_ff)
print('012 voltages: ', V0_abs_vec[ind_min], V1_abs_vec[ind_min], V2_abs_vec[ind_min])

print(V0_im_vec[ind_min])
print(V1_im_vec[ind_min])
print(V2_im_vec[ind_min])

print(V0_re_vec[ind_min])
print(V1_re_vec[ind_min])
print(V2_re_vec[ind_min])


print(V0_im_vec[ind_max])
print(V1_im_vec[ind_max])
print(V2_im_vec[ind_max])

print(V0_re_vec[ind_max])
print(V1_re_vec[ind_max])
print(V2_re_vec[ind_max])

plt.scatter(V1_abs_vec, V2_abs_vec, s=1)
plt.xlabel('|V+|')
plt.ylabel('|V-|')
plt.show()


