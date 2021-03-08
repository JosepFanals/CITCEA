import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


Zf = 0.00 + 0.10 * 1j  # fault impedance
Z1 = 0.01 + 0.10 * 1j  # Za in the drawings
Z2 = 0.01 + 0.05 * 1j  # Zth in the drawings
Imax = 1
Vth_1 = 1  # positive sequence Thévenin voltage

a = np.exp(1j * 120 * np.pi / 180)
lam_1 = 1  # weight to positive sequence
lam_2 = 1  # weight to negative sequence

I1_re = 0.3
I1_im = 0.0
I2_re = - 0.2
I2_im = 0.0

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
    V0 = 0  # balanced
    # V0 = - Z2 / (3 * Zf + 3 * Z2) * (Vth_1 + (x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z2)  # LG
    # V0 = 0  # LL
    # V0 = Z2 / (3 * Z2 + 6 * Zf) * ((x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z2 + Vth_1)  # LLG
    
    return V0

def V1(x):
    V1 = 1 / (Zf + Z2) * (Vth_1 * Zf + (x[0] + 1j * x[1]) * (Z1 * Zf + Z1 * Z2 + Zf * Z2))  # balanced
    # V1 = (x[0] + 1j * x[1]) * (Z1 + Z2) + Vth_1 - Z2 / (3 * Zf + 3 * Z2) * ((x[2] + 1j * x[3]) * Z2 + (x[0] + 1j * x[1]) * Z2 + Vth_1)  # LG
    # V1 = Vth_1 + (x[0] + 1j * x[1]) * Z1 + (x[0] + 1j * x[1]) * Z2 - Z2 / (2 * Z2 + Zf) * (Vth_1 + (x[0] + 1j * x[1]) * Z2 - (x[2] + 1j * x[3]) * Z2)  # LL
    # V1 = (x[0] + 1j * x[1]) * Z1 + (Z2 + 3 * Zf) / (3 * Z2 + 6 * Zf) * ((x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z2 + Vth_1)  # LLG
    
    return V1

def V2(x):
    V2 = 1 / (Zf + Z2) * ((x[2] + 1j * x[3]) * (Z2 * Zf + Z1 * Z2 + Z1 * Zf))  # balanced
    # V2 = (x[2] + 1j * x[3]) * (Z1 + Z2) - Z2 / (3 * Zf + 3 * Z2) * ((x[2] + 1j * x[3]) * Z2 + (x[0] + 1j * x[1]) * Z2 + Vth_1)  # LG
    # V2 = Vth_1 + (x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z1 - (Z2 + Zf) / (2 * Z2 + Zf) * (Vth_1 + (x[0] + 1j * x[1]) * Z2 - (x[2] + 1j * x[3]) * Z2)  # LL
    # V2 = (x[2] + 1j * x[3]) * Z1 + (Z2 + 3 * Zf) / (3 * Z2 + 6 * Zf) * ((x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z2 + Vth_1)

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


x0 = [I1_re, I1_im, I2_re, I2_im]
bound = (-Imax, Imax)
bnds = (bound, bound, bound, bound)
con1 = {'type': 'ineq', 'fun': g1}
con2 = {'type': 'ineq', 'fun': g2}
con3 = {'type': 'ineq', 'fun': g3}
con4 = {'type': 'eq', 'fun': g4}
con5 = {'type': 'eq', 'fun': g5}
cons = [con1, con2, con3]

sol = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'ftol':1e-8})
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

print('--------')
print('|Vabc| voltages: ', abs(Vabcf))
print('|V012| voltages: ', abs(V012f))

ang_shift = np.angle(Vabcf[0])
Iabcf = Iabcf * np.exp(- 1j * ang_shift)
I012 = Vabc_to_012(Iabcf)
print('Iabc currents: ', Iabcf)
print('I012 currents: ', I012)
print('--------')

print(sol)

