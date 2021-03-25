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

    print(Z2)

    #print('.......')
    #print(Rin)
    #print(Xin)
    #print(RX)

    # Zf = 0.10 + 0.00 * 1j  # fault impedance
    Zf = 0.5
    # Z1 = 0.01 + 0.10 * 1j  # Za in the drawings
    Z1 = Z2
    # Z2 = 0.01 + 0.05 * 1j  # Zth in the drawings
    Imax = 1
    Vth_1 = 1  # positive sequence Thévenin voltage

    a = np.exp(1j * 120 * np.pi / 180)
    lam_1 = 1  # weight to positive sequence
    lam_2 = 1  # weight to negative sequence

    # I1_re = 0.3
    # I1_im = 0.0
    # I2_re = - 0.2
    # I2_im = 0.0


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
        # V0 = 0  # LL
        V0 = Z2 / (3 * Z2 + 6 * Zf) * ((x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z2 + Vth_1)  # LLG
        
        return V0

    def V1(x):
        # V1 = 1 / (Zf + Z2) * (Vth_1 * Zf + (x[0] + 1j * x[1]) * (Z1 * Zf + Z1 * Z2 + Zf * Z2))  # balanced
        # V1 = (x[0] + 1j * x[1]) * (Z1 + Z2) + Vth_1 - Z2 / (3 * Zf + 3 * Z2) * ((x[2] + 1j * x[3]) * Z2 + (x[0] + 1j * x[1]) * Z2 + Vth_1)  # LG
        # V1 = Vth_1 + (x[0] + 1j * x[1]) * Z1 + (x[0] + 1j * x[1]) * Z2 - Z2 / (2 * Z2 + Zf) * (Vth_1 + (x[0] + 1j * x[1]) * Z2 - (x[2] + 1j * x[3]) * Z2)  # LL
        V1 = (x[0] + 1j * x[1]) * Z1 + (Z2 + 3 * Zf) / (3 * Z2 + 6 * Zf) * ((x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z2 + Vth_1)  # LLG
        
        return V1

    def V2(x):
        # V2 = 1 / (Zf + Z2) * ((x[2] + 1j * x[3]) * (Z2 * Zf + Z1 * Z2 + Z1 * Zf))  # balanced
        # V2 = (x[2] + 1j * x[3]) * (Z1 + Z2) - Z2 / (3 * Zf + 3 * Z2) * ((x[2] + 1j * x[3]) * Z2 + (x[0] + 1j * x[1]) * Z2 + Vth_1)  # LG
        # V2 = Vth_1 + (x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z1 - (Z2 + Zf) / (2 * Z2 + Zf) * (Vth_1 + (x[0] + 1j * x[1]) * Z2 - (x[2] + 1j * x[3]) * Z2)  # LL
        V2 = (x[2] + 1j * x[3]) * Z1 + (Z2 + 3 * Zf) / (3 * Z2 + 6 * Zf) * ((x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z2 + Vth_1)

        return V2

    def Ia_re(x):
        Ia_re = x[0] + x[2]
        return Ia_re

    def Ia_im(x):
        Ia_im = x[1] + x[3]
        return Ia_im

    def Ib_re(x):
        Ib_re = - 1 / 2 * x[0] + np.sqrt(3) / 2 * x[1] - 1 / 2 * x[2] - np.sqrt(3) / 2 * x[3]
        return Ib_re

    def Ib_im(x):
        Ib_im = - 1 / 2 * x[1] - np.sqrt(3) / 2 * x[0] - 1 / 2 * x[3] + np.sqrt(3) / 2 * x[2]
        return Ib_im

    def Ic_re(x):
        Ic_re = - 1 / 2 * x[0] - np.sqrt(3) / 2 * x[1] - 1 / 2 * x[2] + np.sqrt(3) / 2 * x[3]
        return Ic_re

    def Ic_im(x):
        Ic_im = np.sqrt(3) / 2 * x[0] - 1 / 2 * x[1] - np.sqrt(3) / 2 * x[2] - 1 / 2 * x[3]
        return Ic_im

    def objective(x):
        objective = lam_1 * abs(np.abs(V1(x)) - 1) + lam_2 * abs(np.abs(V2(x)) - 0) 
        return objective

    def g1(x):
        return Imax - np.sqrt(Ia_re(x) ** 2 + Ia_im(x) ** 2)

    def g2(x):
        return Imax - np.sqrt(Ib_re(x) ** 2 + Ib_im(x) ** 2)

    def g3(x):
        return Imax - np.sqrt(Ic_re(x) ** 2 + Ic_im(x) ** 2)

    def g4(x):
        return Ia_re(x) + Ib_re(x) + Ic_re(x)

    def g5(x):
        return Ia_im(x) + Ib_im(x) + Ic_im(x)

    def ang_Vabc(x):
        V012_ang = np.array([V0(x), V1(x), V2(x)])
        Vabc_ang = V012_to_abc(V012_ang)
        ang_Va = np.angle(Vabc_ang[0])
        ang_Vb = np.angle(Vabc_ang[1])
        ang_Vc = np.angle(Vabc_ang[2])
        return [ang_Va, ang_Vb, ang_Vc]

    def g9(x):
        V012_r = [V0(x), V1(x), V2(x)]
        Vabc_r = V012_to_abc(V012_r)
        I012_r = [0, x[0] + 1j * x[1], x[2] + 1j * x[3]]
        Iabc_r = V012_to_abc(I012_r)
        return np.real(Vabc_r[0] * np.conj(Iabc_r[0]) + Vabc_r[1] * np.conj(Iabc_r[1]) + Vabc_r[2] * np.conj(Iabc_r[2]))
        

    # x0 = [I1_re, I1_im, I2_re, I2_im]
    bound = (-Imax, Imax)
    bnds = (bound, bound, bound, bound)
    con1 = {'type': 'ineq', 'fun': g1}
    con2 = {'type': 'ineq', 'fun': g2}
    con3 = {'type': 'ineq', 'fun': g3}
    con4 = {'type': 'eq', 'fun': g4}
    con5 = {'type': 'eq', 'fun': g5}
    con9 = {'type': 'eq', 'fun': g9}
    cons = [con1, con2, con3]

    sol = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'ftol':1e-9})
    Iopt = sol.x
    I0f = 0
    I1f = Iopt[0] + 1j * Iopt[1]
    I2f = Iopt[2] + 1j * Iopt[3]
    I012f = np.array([I0f, I1f, I2f])
    Iabcf = V012_to_abc(I012f)

    V0f = V0(Iopt)
    V1f = V1(Iopt)
    V2f = V2(Iopt)
    V012f = np.array([V0f, V1f, V2f])
    Vabcf = V012_to_abc(V012f)

    # ang_shift = np.angle(Vabcf[0])
    # # print(Iabcf)
    # Iabcf = Iabcf * np.exp(+ 1j * ang_shift)
    # Vabcf = Vabcf * np.exp(+ 1j * ang_shift)
    # # print(Iabcf)
    # I012f = Vabc_to_012(Iabcf)
    # V012f = Vabc_to_012(Vabcf)

    I012f[1] = I012f[1] * np.exp(- 1j * np.angle(V012f[1]))  # added
    I012f[2] = I012f[2] * np.exp(- 1j * np.angle(V012f[2]))
    V012f[1] = V012f[1] * np.exp(- 1j * np.angle(V012f[1]))
    V012f[2] = V012f[2] * np.exp(- 1j * np.angle(V012f[2]))

    I1_vec.append(I012f[1])
    I2_vec.append(I012f[2])
    V1_vec.append(V012f[1])
    V2_vec.append(V012f[2])
    ff_vec.append(sol.fun)

    #print('Iabc currents: ', Iabcf)
    #print('I012 currents: ', I012)
    #print('--------')

    # print(sol)
    # print(sol.fun)
    print('Voltages V012: ', V012f)
    print('Voltages Vabc: ', Vabcf)
    print('Currents I012: ', I012f)
    print('Currents Iabc: ', Iabcf)
    S_tot = Vabcf[0] * np.conj(Iabcf[0]) + Vabcf[1] * np.conj(Iabcf[1]) + Vabcf[2] * np.conj(Iabcf[2])
    S_tot2 = V012f[1] * np.conj(I012f[1]) + V012f[2] * np.conj(I012f[2])
    P_vec.append(np.real(S_tot))
    Q_vec.append(np.imag(S_tot))
    print(sol.success)
    print(S_tot)
    print(3 * S_tot2)
    print((np.angle(V012f[1]) - np.angle(I012f[1])) * 180 / np.pi)
    print((np.angle(V012f[2]) - np.angle(I012f[2])) * 180 / np.pi)
    print('----------')


# ///////////// plots //////////////

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

make_csv(RX_vec, np.real(I1_vec), a_vec, 'Optimal/Data/RX/I1_re_LLG.csv')
make_csv(RX_vec, np.imag(I1_vec), a_vec, 'Optimal/Data/RX/I1_im_LLG.csv')
make_csv(RX_vec, np.real(I2_vec), a_vec, 'Optimal/Data/RX/I2_re_LLG.csv')
make_csv(RX_vec, np.imag(I2_vec), a_vec, 'Optimal/Data/RX/I2_im_LLG.csv')
make_csv(RX_vec, ff_vec, a_vec, 'Optimal/Data/RX/ff_LLG.csv')

make_csv(RX_vec, np.abs(V1_vec), a_vec, 'Optimal/Data/RX/V1_LLG.csv')
make_csv(RX_vec, np.abs(V2_vec), a_vec, 'Optimal/Data/RX/V2_LLG.csv')
make_csv(RX_vec, ff_vec, a_vec, 'Optimal/Data/RX/ffG_LLG.csv')


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