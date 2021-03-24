import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

I1_vec = []
I2_vec = []
ff_vec = []
RX_vec = []
V1_vec = []
V2_vec = []
P_vec = []
Q_vec = []

lim1_RX = 5
lim2_RX = 0.1
difff = lim1_RX - lim2_RX
n_punts = 100
increment = difff / n_punts
Zthmod = np.sqrt(0.01 ** 2 + 0.05 ** 2)

# initialize values
I1_re = 0.0
I1_im = 0.0
I2_re = 0.0
I2_im = 0.0
x0 = [I1_re, I1_im, I2_re, I2_im]

for kk in range(n_punts):

    RX = lim2_RX + increment * kk
    Xin = np.sqrt(Zthmod ** 2 / (1 + RX ** 2))
    Rin = RX * Xin

    Z2 = Rin + Xin * 1j
    RX_vec.append(RX)

    Zf = 0.03
    Z1 = Z2
    Imax = 1
    Vth_1 = 1  # positive sequence Thévenin voltage

    a = np.exp(1j * 120 * np.pi / 180)
    lam_1 = 1  # weight to positive sequence
    lam_2 = 1  # weight to negative sequence

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

        Vabc = np.dot(T, V012)
        return Vabc

    def V0(x):
        # V0 = 0  # balanced
        # V0 = - Z2 / (3 * Zf + 3 * Z2) * (Vth_1 + (x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z2)  # LG
        V0 = 0  # LL
        # V0 = Z2 / (3 * Z2 + 6 * Zf) * ((x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z2 + Vth_1)  # LLG
        
        return V0

    def V1(x):
        # V1 = 1 / (Zf + Z2) * (Vth_1 * Zf + (x[0] + 1j * x[1]) * (Z1 * Zf + Z1 * Z2 + Zf * Z2))  # balanced
        # V1 = (x[0] + 1j * x[1]) * (Z1 + Z2) + Vth_1 - Z2 / (3 * Zf + 3 * Z2) * ((x[2] + 1j * x[3]) * Z2 + (x[0] + 1j * x[1]) * Z2 + Vth_1)  # LG
        V1 = Vth_1 + (x[0] + 1j * x[1]) * Z1 + (x[0] + 1j * x[1]) * Z2 - Z2 / (2 * Z2 + Zf) * (Vth_1 + (x[0] + 1j * x[1]) * Z2 - (x[2] + 1j * x[3]) * Z2)  # LL
        # V1 = (x[0] + 1j * x[1]) * Z1 + (Z2 + 3 * Zf) / (3 * Z2 + 6 * Zf) * ((x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z2 + Vth_1)  # LLG
        
        return V1

    def V2(x):
        # V2 = 1 / (Zf + Z2) * ((x[2] + 1j * x[3]) * (Z2 * Zf + Z1 * Z2 + Z1 * Zf))  # balanced
        # V2 = (x[2] + 1j * x[3]) * (Z1 + Z2) - Z2 / (3 * Zf + 3 * Z2) * ((x[2] + 1j * x[3]) * Z2 + (x[0] + 1j * x[1]) * Z2 + Vth_1)  # LG
        V2 = Vth_1 + (x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z1 - (Z2 + Zf) / (2 * Z2 + Zf) * (Vth_1 + (x[0] + 1j * x[1]) * Z2 - (x[2] + 1j * x[3]) * Z2)  # LL
        # V2 = (x[2] + 1j * x[3]) * Z1 + (Z2 + 3 * Zf) / (3 * Z2 + 6 * Zf) * ((x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z2 + Vth_1)

        return V2

    def I1_grid_code(V012, limits, fac):
        V1 = np.abs(V012[1])
        v1_ang = np.angle(V012[1])
        if limits == False:
            if V1 >= 0.9:
                I1 = 0
            elif V1 >= 0.5:
                I1 = ksp * (0.9 - V1)
            else:
                I1 = 1
        else:
            if V1 >= 0.9:
                I1 = 0
            elif V1 >= 0.5:
                I1 = fac * ksp * (0.9 - V1)
            else:
                I1 = fac * 1
    
        return I1 * -1j * np.exp(1j * v1_ang)

    def I2_grid_code(V012, limits, fac):
        V2 = np.abs(V012[2])
        v2_ang = np.angle(V012[2])
        if limits == False:
            if V2 <= 0.1:
                I2 = 0
            elif V2 < 0.5:
                I2 = ksp * (V2 - 0.1)
            else:
                I2 = 1
        else:
            if V2 <= 0.1:
                I2 = 0
            elif V2 < 0.5:
                I2 = fac * ksp * (V2 - 0.1)
            else:
                I2 = fac * 1
        
        return I2 * 1j * np.exp(1j * v2_ang)

    x = [0, 0, 0, 0]
    V012f = np.array([V0(x), V1(x), V2(x)])
    I012f = np.array([0, x[0] + 1j * x[1], x[2] + 1j * x[3]])
    Iabcf = V012_to_abc(I012f)
    Vabcf = V012_to_abc(V012f)


    # GRID CODE CURRENT COMPUTATION

    ksp = 2.5
    V1_new = V012f[1]
    V1_old = 0
    V2_new = V012f[2]
    V2_old = 0
    tol = 1e-5
    compt = 0
    compt_lim = 100

    while (abs(V1_new - V1_old) > tol or abs(V2_new - V2_old) > tol) and compt < compt_lim:

        V1_old = V1_new
        V2_old = V2_new
        limits = False

        I1_gc = I1_grid_code(V012f, limits, 0)
        I2_gc = I2_grid_code(V012f, limits, 0)
        I012_gc = np.array([0, I1_gc, I2_gc])
        Iabc_gc = V012_to_abc(I012_gc)

        Iabc_max = max(abs(Iabc_gc))
        x_gc = [np.real(I1_gc), np.imag(I1_gc), np.real(I2_gc), np.imag(I2_gc)]
        V012_gc = np.array([V0(x_gc), V1(x_gc), V2(x_gc)])

        if Iabc_max > Imax:
            limits = True
            fac = 1
            while Iabc_max > Imax or Iabc_max < 0.99 * Imax:
                if Iabc_max < Imax:
                    fac += 0.0001 
                else:
                    fac -= 0.0001 

                I1_gc = I1_grid_code(V012f, limits, fac)
                I2_gc = I2_grid_code(V012f, limits, fac)

                I012_gc = np.array([0, I1_gc, I2_gc])
                Iabc_gc = V012_to_abc(I012_gc)

                Iabc_max = max(abs(Iabc_gc))
                x_gc = [np.real(I1_gc), np.imag(I1_gc), np.real(I2_gc), np.imag(I2_gc)]
                V012_gc = np.array([V0(x_gc), V1(x_gc), V2(x_gc)])


        I012_gc = np.array([0, I1_gc, I2_gc])
        Iabc_gc = V012_to_abc(I012_gc)
        Iabc_max = max(abs(Iabc_gc))
        x_gc = [np.real(I1_gc), np.imag(I1_gc), np.real(I2_gc), np.imag(I2_gc)]
        V012_gc = np.array([V0(x_gc), V1(x_gc), V2(x_gc)])

        V012f = V012_gc
        V1_new = abs(V012_gc[1])
        V2_new = abs(V012_gc[2])

        compt += 1
        print(compt)

    I1_vec.append((x_gc[0] + 1j * x_gc[1]) * np.exp(- 1j * np.angle(V012_gc[1])))
    I2_vec.append((x_gc[2] + 1j * x_gc[3]) * np.exp(- 1j * np.angle(V012_gc[2])))
    V1_vec.append(V012_gc[1])
    V2_vec.append(V012_gc[2])
    ff_vec.append(abs(1 - abs(V012_gc[1]) + abs(0 - abs(V012_gc[2]))))
    # I1_vec.append((x_gc[0] + 1j * x_gc[1]) * np.exp(1j * np.angle(V1_vec)))
    # I2_vec.append((x_gc[2] + 1j * x_gc[3]) * np.exp(1j * np.angle(V2_vec)))
    # print(np.angle(V1_vec[-1]) - np.angle(I1_vec[-1]))
    # print(I1_vec[-1])
    # print(V1_vec[-1])

def make_csv(x_vec, y_vec, a_vec, file_name):
    df = pd.DataFrame(data=[x_vec, y_vec, a_vec]).T
    df.columns = ['x', 'y', 'label']
    df.to_csv(file_name, index=False)

def make3_csv(x_vec, y_vec, z_vec, file_name):
    df = pd.DataFrame(data=[x_vec, y_vec, z_vec]).T
    df.columns = ['x', 'y', 'z']
    df.to_csv(file_name, index=False)


zero_vec = np.full(len(RX_vec), 0)
one_vec = np.full(len(RX_vec), 5)
a_vec = np.full(len(RX_vec), 'a')
b_vec = 'b'

make_csv(RX_vec, np.real(I1_vec), a_vec, 'Optimal/Data/gridcode/GI1_re_LL.csv')
make_csv(RX_vec, np.imag(I1_vec), a_vec, 'Optimal/Data/gridcode/GI1_im_LL.csv')
make_csv(RX_vec, np.real(I2_vec), a_vec, 'Optimal/Data/gridcode/GI2_re_LL.csv')
make_csv(RX_vec, np.imag(I2_vec), a_vec, 'Optimal/Data/gridcode/GI2_im_LL.csv')
make_csv(RX_vec, ff_vec, a_vec, 'Optimal/Data/gridcode/Gff_LL.csv')

make3_csv(RX_vec, one_vec, np.abs(V1_vec), 'Optimal/Data/gridcode/GV1_LL.csv')
make3_csv(zero_vec, RX_vec, np.abs(V2_vec), 'Optimal/Data/gridcode/GV2_LL.csv')
make3_csv(RX_vec, RX_vec, ff_vec, 'Optimal/Data/gridcode/GffG_LL.csv')


fig, axs = plt.subplots(3,2)
fig.suptitle('Vertically and horizontally stacked subplots')
axs[0,0].plot(RX_vec, np.real(I1_vec))
axs[0,1].plot(RX_vec, np.imag(I1_vec))
axs[1,0].plot(RX_vec, np.real(I2_vec))
axs[1,1].plot(RX_vec, np.imag(I2_vec))
axs[2,0].plot(RX_vec, ff_vec)

plt.ylim(-1, 1)
plt.xlim(lim2_RX, lim1_RX)

axs[0,0].set_xlim([lim2_RX, lim1_RX])
axs[0,0].set_ylim([-1, 1])

axs[0,1].set_xlim([lim2_RX, lim1_RX])
axs[0,1].set_ylim([-1, 1])

axs[1,0].set_xlim([lim2_RX, lim1_RX])
axs[1,0].set_ylim([-1, 1])

axs[1,1].set_xlim([lim2_RX, lim1_RX])
axs[1,1].set_ylim([-1, 1])

axs[2,0].set_xlim([lim2_RX, lim1_RX])
axs[2,0].set_ylim([0, 1])

plt.show()