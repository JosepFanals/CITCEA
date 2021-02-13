import numpy as np


def V012_to_abc(V012):
    T = np.zeros((3,3), dtype=complex)
    T[0,0] = 1 / 3
    T[0,1] = 1 / 3
    T[0,2] = 1 / 3

    T[1,0] = 1 / 3
    T[1,1] = 1 / 3 * np.exp(1j * 2 * np.pi / 3)
    T[1,2] = 1 / 3 * np.exp(-1j * 2 * np.pi / 3)

    T[2,0] = 1 / 3
    T[2,1] = 1 / 3 * np.exp(-1j * 2 * np.pi / 3)
    T[2,2] = 1 / 3 * np.exp(1j * 2 * np.pi / 3)

    Tinv = np.linalg.inv(T)
    Vabc = np.dot(Tinv, V012)
    return Vabc


def Vabc_to_012(Vabc):
    T = np.zeros((3,3), dtype=complex)
    T[0,0] = 1 / 3
    T[0,1] = 1 / 3
    T[0,2] = 1 / 3

    T[1,0] = 1 / 3
    T[1,1] = 1 / 3 * np.exp(1j * 2 * np.pi / 3)
    T[1,2] = 1 / 3 * np.exp(-1j * 2 * np.pi / 3)

    T[2,0] = 1 / 3
    T[2,1] = 1 / 3 * np.exp(-1j * 2 * np.pi / 3)
    T[2,2] = 1 / 3 * np.exp(1j * 2 * np.pi / 3)

    V012 = np.dot(T, Vabc)
    return V012


R = 0.01
X = 0.1

Vgp = 0.8
Vgn = 0.1 

Ipre = 0.0
Ipim = 1.4
Inre = 0.9
Inim = 0.0

Vcp = R * Ipre + X * Ipim + np.sqrt(Vgp ** 2 - (X * Ipre - R * Ipim) ** 2)
Vcn = R * Inre - X * Inim + np.sqrt(Vgn ** 2 - (R * Inim + X * Inre) ** 2)
print(Vcp, Vcn)
 

