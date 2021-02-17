# matrix can be wrongly defined!!

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


Vz1 = 0.9 * np.exp(1j * 0.1j)
Vz2 = 0.1 * np.exp(1j * 0.2)
Vz0 = 0.0
Vz012 = np.array([Vz0, Vz1, Vz2])
Vzabc = V012_to_abc(Vz012)
print(Vzabc)
Zz = 0.01 + 0.1 * 1j

Il1 = - 0.7 * 1j
Il2 = 0.3 * 1j
Il0 = 0
Il012 = np.array([Il0, Il1, Il2])

Vc1 = Vz1 + Zz * Il1
Vc2 = Vz2 + Zz * Il2
Vc0 = 0
Vc012 = np.array([Vc0, Vc1, Vc2])
print(Vc012)

Vcabc = V012_to_abc(Vc012)
print(Vcabc)
Ilabc = V012_to_abc(Il012)
print(abs(Ilabc))