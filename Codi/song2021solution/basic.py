# song2021solution, system

import numpy as np

# DEFINITIONS 
uth = 1
zth = 0.01 + 0.1 * 1j
pref = 0.5
uref = 1
zvsc = 0.1j


def f1(dd, Q):
    ff = 1.74 * np.cos(dd) + 9.9 * np.sin(dd) - Q * np.sin(dd) - 0.99
    return ff


def f2(dd, Q):
    ff = 9.9 * np.cos(dd) + 0.24 * np.sin(dd) - Q * np.cos(dd) - 9.9
    return ff



# CALCULATION
tol = 1e-8
dd = 0
Q = 0
n_iter = 10

for i in range(n_iter):
    J = np.array([[(f1(dd + tol, Q) - f1(dd, Q)) / tol, (f1(dd, Q + tol) - f1(dd, Q)) / tol],[(f2(dd + tol, Q) - f2(dd, Q)) / tol, (f2(dd, Q + tol) - f2(dd, Q)) / tol]])
    Jinv = np.linalg.inv(J)
    ff1 = f1(dd, Q)
    ff2 = f2(dd, Q)

    dd += - (Jinv[0,0] * ff1 + Jinv[0,1] * ff2)
    Q += -(Jinv[1,0] * ff1 + Jinv[1,1] * ff2)

    print('N: ', i, 'delta: ', dd, 'Q: ', Q)


Io = 1.25 * (np.cos(dd) + np.sin(dd) * 1j)
Iv = 0.5 * (np.cos(dd) + np.sin(dd) * 1j) + Q * (np.cos(dd-np.pi/2) + np.sin(dd-np.pi/2) * 1j)
Vv = 1 * (np.cos(dd) + np.sin(dd) * 1j) + Iv * zvsc


print('Io: ', Io)
print('Iv: ', Iv)
print('VV: ', Vv)