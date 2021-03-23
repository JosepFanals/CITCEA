import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd


Zf = 0.03 + 0.00 * 1j  # fault impedance
Z1 = 0.01 + 0.10 * 1j  # Za in the drawings
Z2 = 0.01 + 0.05 * 1j  # Zth in the drawings
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
    V0 = - Z2 / (3 * Zf + 3 * Z2) * (Vth_1 + (x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z2)  # LG
    # V0 = 0  # LL
    # V0 = Z2 / (3 * Z2 + 6 * Zf) * ((x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z2 + Vth_1)  # LLG
    
    return V0

def V1(x):
    # V1 = 1 / (Zf + Z2) * (Vth_1 * Zf + (x[0] + 1j * x[1]) * (Z1 * Zf + Z1 * Z2 + Zf * Z2))  # balanced
    V1 = (x[0] + 1j * x[1]) * (Z1 + Z2) + Vth_1 - Z2 / (3 * Zf + 3 * Z2) * ((x[2] + 1j * x[3]) * Z2 + (x[0] + 1j * x[1]) * Z2 + Vth_1)  # LG
    # V1 = Vth_1 + (x[0] + 1j * x[1]) * Z1 + (x[0] + 1j * x[1]) * Z2 - Z2 / (2 * Z2 + Zf) * (Vth_1 + (x[0] + 1j * x[1]) * Z2 - (x[2] + 1j * x[3]) * Z2)  # LL
    # V1 = (x[0] + 1j * x[1]) * Z1 + (Z2 + 3 * Zf) / (3 * Z2 + 6 * Zf) * ((x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z2 + Vth_1)  # LLG
    
    return V1

def V2(x):
    # V2 = 1 / (Zf + Z2) * ((x[2] + 1j * x[3]) * (Z2 * Zf + Z1 * Z2 + Z1 * Zf))  # balanced
    V2 = (x[2] + 1j * x[3]) * (Z1 + Z2) - Z2 / (3 * Zf + 3 * Z2) * ((x[2] + 1j * x[3]) * Z2 + (x[0] + 1j * x[1]) * Z2 + Vth_1)  # LG
    # V2 = Vth_1 + (x[0] + 1j * x[1]) * Z2 + (x[2] + 1j * x[3]) * Z1 - (Z2 + Zf) / (2 * Z2 + Zf) * (Vth_1 + (x[0] + 1j * x[1]) * Z2 - (x[2] + 1j * x[3]) * Z2)  # LL
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
print(abs(V012f))

# ang_va = np.angle(Vabcf[0])
# Vabcf = Vabcf * np.exp(- 1j * ang_va)
# Iabcf = Iabcf * np.exp(- 1j * ang_va)
# V012f = Vabc_to_012(Vabcf)
# I012f = Vabc_to_012(Iabcf)



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
            # print(fac)


    I012_gc = np.array([0, I1_gc, I2_gc])
    Iabc_gc = V012_to_abc(I012_gc)
    Iabc_max = max(abs(Iabc_gc))
    x_gc = [np.real(I1_gc), np.imag(I1_gc), np.real(I2_gc), np.imag(I2_gc)]
    V012_gc = np.array([V0(x_gc), V1(x_gc), V2(x_gc)])

    V012f = V012_gc
    # print(abs(V012_gc))
    print(Iabc_max)

    V1_new = abs(V012_gc[1])
    V2_new = abs(V012_gc[2])

    # print(compt)
    compt += 1

