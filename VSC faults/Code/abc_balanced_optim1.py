import numpy as np
from scipy.optimize import minimize


Zf = 0.01 + 0.1 * 1j
Z1 = 0.02 + 0.04 * 1j
Z2 = 0.01 + 0.06 * 1j 
Imax = 1
a = np.exp(1j * 120 * np.pi / 180)
b = a ** 2
a_re = np.real(a)
a_im = np.imag(a)
b_re = np.real(b)
b_im = np.imag(b)

Vth_a = 1
Vth_b = 1 * b
Vth_c = 1 * a

Vth_a_re = np.real(Vth_a)
Vth_a_im = np.imag(Vth_a)
Vth_b_re = np.real(Vth_b)
Vth_b_im = np.imag(Vth_b)
Vth_c_re = np.real(Vth_c)
Vth_c_im = np.imag(Vth_c)

Yx = 1 / (Zf + Z2)
Zx = (Z1 * Z2 + Z2 * Zf + Zf * Z1)
K = Yx * Zf
T = Yx * Zx
K_re = np.real(K)
K_im = np.imag(K)
T_re = np.real(T)
T_im = np.imag(T)

lam_1 = 1  # weight to positive sequence
lam_2 = 1  # weight to negative sequence

Ia_re0 = 0.1
Ia_im0 = 0.1
Ib_re0 = 0.05
Ib_im0 = 0.05
Ic_re0 = 0.05
Ic_im0 = 0.05

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

def Va_re(x):
    Va_re = K_re * Vth_a_re - K_im * Vth_a_im + T_re * x[0] - T_im * x[1]
    return Va_re

def Va_im(x):
    Va_im = K_re * Vth_a_im + K_im * Vth_a_re + T_re * x[1] + T_im * x[0]
    return Va_im

def Vb_re(x):
    Vb_re = K_re * Vth_b_re - K_im * Vth_b_im + T_re * x[2] - T_im * x[3]
    return Vb_re

def Vb_im(x):
    Vb_im = K_re * Vth_b_im + K_im * Vth_b_re + T_re * x[3] + T_im * x[2]
    return Vb_im

def Vc_re(x):
    Vc_re = K_re * Vth_c_re - K_im * Vth_c_im + T_re * x[4] - T_im * x[5]
    return Vc_re

def Vc_im(x):
    Vc_im = K_re * Vth_c_im + K_im * Vth_c_re + T_re * x[5] + T_im * x[4]
    return Vc_im

def V1_re(x):
    V1_re = 1 / 3 * (Va_re(x) + a_re * Vb_re(x) - a_im * Vb_im(x) + b_re * Vc_re(x) - b_im * Vc_im(x))
    return V1_re

def V1_im(x): 
    V1_im = 1 / 3 * (Va_im(x) + a_re * Vb_im(x) + a_im * Vb_re(x) + b_re * Vc_im(x) + b_im * Vc_re(x))
    return V1_im

def V2_re(x):
    V2_re = 1 / 3 * (Va_re(x) + b_re * Vb_re(x) - b_im * Vb_im(x) + a_re * Vc_re(x) - a_im * Vc_im(x))
    return V2_re

def V2_im(x): 
    V2_im = 1 / 3 * (Va_im(x) + b_re * Vb_im(x) + b_im * Vb_re(x) + a_re * Vc_im(x) + a_im * Vc_re(x))
    return V2_im

def objective(x):
    objective = lam_1 * abs(np.sqrt(V1_re(x) ** 2 + V1_im(x) ** 2) - 1) + lam_2 * abs(np.sqrt(V2_re(x) ** 2 + V2_im(x) ** 2) - 0) 
    return objective

def g1(x):
    return Imax - np.sqrt(x[0] ** 2 + x[1] ** 2)

def g2(x):
    return Imax - np.sqrt(x[2] ** 2 + x[3] ** 2)

def g3(x):
    return Imax - np.sqrt(x[4] ** 2 + x[5] ** 2)

def g4(x):
    return sum(x)


x0 = [Ia_re0, Ia_im0, Ib_re0, Ib_im0, Ic_re0, Ic_im0]
bound = (-Imax, Imax)
bnds = (bound, bound, bound, bound, bound, bound)
con1 = {'type': 'ineq', 'fun': g1}
con2 = {'type': 'ineq', 'fun': g2}
con3 = {'type': 'ineq', 'fun': g3}
con4 = {'type': 'eq', 'fun': g4}
cons = [con1, con2, con3, con4]

sol = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
Iopt = sol.x
Iaf = Iopt[0] + 1j * Iopt[1]
Ibf = Iopt[2] + 1j * Iopt[3]
Icf = Iopt[4] + 1j * Iopt[5]

Vaf = Va_re(Iopt) + 1j * Va_im(Iopt)
Vbf = Vb_re(Iopt) + 1j * Vb_im(Iopt)
Vcf = Vc_re(Iopt) + 1j * Vc_im(Iopt)

ang_shift = np.angle(Vaf)
Iaf = Iaf * np.exp(- 1j * ang_shift)
Ibf = Ibf * np.exp(- 1j * ang_shift)
Icf = Icf * np.exp(- 1j * ang_shift)
Iabc = np.array([Iaf, Ibf, Icf])
I012 = Vabc_to_012(Iabc)
print(I012)