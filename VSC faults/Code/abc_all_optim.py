import numpy as np
from scipy.optimize import minimize


Zf = 0.00 + 0.10 * 1j  # fault impedance
Z1 = 0.01 + 0.10 * 1j  # Za in the drawings
Z2 = 0.01 + 0.05 * 1j  # Zth in the drawings
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

def Va(x):
    # Va = 1 / (Zf + Z2) * (Vth_a * Zf + (x[0] + 1j * x[1]) * (Z1 * Z2 + Z2 * Zf + Zf * Z1))  # balanced
    # Va = 1 / (Zf + Z2) * (Vth_a * Zf + (x[0] + 1j * x[1]) * (Z1 * Z2 + Z2 * Zf + Zf * Z1))  # LG
    Va = Vth_a + (x[0] + 1j * x[1]) * (Z1 + Z2)  # LL
    # Va = Vth_a + (x[0] + 1j * x[1]) * (Z1 + Z2)  # LLG

    return Va

def Vb(x):
    # Vb = 1 / (Zf + Z2) * (Vth_b * Zf + (x[2] + 1j * x[3]) * (Z1 * Z2 + Z2 * Zf + Zf * Z1))  # balanced
    # Vb = Vth_b + (x[2] + 1j * x[3]) * (Z1 + Z2)  # LG
    Vb = (x[2] + 1j * x[3]) * Z1 + 1 / ((Zf + Z2) * (Zf + 2 * Z2)) * ((x[2] + 1j * x[3]) * (Z2 * Zf * Zf + 2 * Z2 * Z2 * Zf + Z2 * Z2 * Z2) + (x[4] + 1j * x[5]) * (Z2 * Z2 * Zf + Z2 * Z2 * Z2) + Vth_b * (Zf * Zf + 2 * Zf * Z2 + Z2 * Z2) + Vth_c * (Z2 * Zf + Z2 * Z2))  # LL
    # Vb = (x[2] + 1j * x[3]) * Z1 + (Z2 * Zf * (x[2] + 1j * x[3] + x[4] + 1j * x[5]) + Zf * (Vth_b + Vth_c)) / (2 * Zf + Z2)  # LLG

    return Vb

def Vc(x):
    # Vc = 1 / (Zf + Z2) * (Vth_c * Zf + (x[4] + 1j * x[5]) * (Z1 * Z2 + Z2 * Zf + Zf * Z1))  # balanced
    # Vc = Vth_c + (x[4] + 1j * x[5]) * (Z1 + Z2)  # LG
    Vc = (x[4] + 1j * x[5]) * Z1 + 1 / (Zf + 2 * Z2) * ((x[4] + 1j * x[5]) * (Z2 * (Zf + Z2)) + Vth_c * (Zf + Z2) + (x[2] + 1j * x[3]) * Z2 * Z2 + Vth_b * Z2)  # LL
    # Vc = (x[4] + 1j * x[5]) * Z1 + (Z2 * Zf * (x[2] + 1j * x[3] + x[4] + 1j * x[5]) + Zf * (Vth_b + Vth_c)) / (2 * Zf + Z2)  # LLG

    return Vc

def V0(x):
    V0 = 1 / 3 * (Va(x) + Vb(x) + Vc(x))
    return V0

def V1(x):
    V1 = 1 / 3 * (Va(x) + a * Vb(x) + a ** 2 * Vc(x))
    return V1

def V2(x):
    V2 = 1 / 3 * (Va(x) + a ** 2 * Vb(x) + a * Vc(x))
    return V2

def objective(x):
    objective = lam_1 * abs(np.abs(V1(x)) - 1) + lam_2 * abs(np.abs(V2(x)) - 0) 
    return objective

def g1(x):
    return Imax - np.sqrt(x[0] ** 2 + x[1] ** 2)

def g2(x):
    return Imax - np.sqrt(x[2] ** 2 + x[3] ** 2)

def g3(x):
    return Imax - np.sqrt(x[4] ** 2 + x[5] ** 2)

def g4(x):
    return x[0] + x[2] + x[4]

def g5(x):
    return x[1] + x[3] + x[5]


x0 = [Ia_re0, Ia_im0, Ib_re0, Ib_im0, Ic_re0, Ic_im0]
bound = (-Imax, Imax)
bnds = (bound, bound, bound, bound, bound, bound)
con1 = {'type': 'ineq', 'fun': g1}
con2 = {'type': 'ineq', 'fun': g2}
con3 = {'type': 'ineq', 'fun': g3}
con4 = {'type': 'eq', 'fun': g4}
con5 = {'type': 'eq', 'fun': g5}
cons = [con1, con2, con3, con4, con5]

# sol = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
sol = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'ftol':1e-10})
Iopt = sol.x
Iaf = Iopt[0] + 1j * Iopt[1]
Ibf = Iopt[2] + 1j * Iopt[3]
Icf = Iopt[4] + 1j * Iopt[5]

V0f = V0(Iopt)
V1f = V1(Iopt)
V2f = V2(Iopt)
V012f = np.array([V0f, V1f, V2f])

Vaf = Va(Iopt)
Vbf = Vb(Iopt)
Vcf = Vc(Iopt)
Vabcf = np.array([Vaf, Vbf, Vcf])
print('--------')
print('|Vabc| voltages: ', abs(Vabcf))
print('|V012| voltages: ', abs(V012f))

ang_shift = np.angle(Vaf)
Iaf = Iaf * np.exp(- 1j * ang_shift)
Ibf = Ibf * np.exp(- 1j * ang_shift)
Icf = Icf * np.exp(- 1j * ang_shift)
Iabc = np.array([Iaf, Ibf, Icf])
I012 = Vabc_to_012(Iabc)
print('Iabc currents: ', Iabc)
print('I012 currents: ', I012)
print('--------')

print(sol)