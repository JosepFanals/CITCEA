import numpy as np


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


def fIa(Ipre, Ipim, Inre, Inim):
    valor = Ipre ** 2 + Inre ** 2 + Ipim ** 2 + Inim ** 2 + 2 * Ipre * Inre - 2 * Ipim * Inim
    return np.sqrt(valor)


def fIb(Ipre, Ipim, Inre, Inim):
    valor = Ipre ** 2 + Inre ** 2 + Ipim ** 2 + Inim ** 2 - Ipre * Inre + np.sqrt(3) * Ipre * Inim + np.sqrt(3) * Inre * Ipim + Ipim * Inim
    return np.sqrt(valor)


def fIc(Ipre, Ipim, Inre, Inim):
    valor = Ipre ** 2 + Inre ** 2 + Ipim ** 2 + Inim ** 2 - Ipre * Inre - np.sqrt(3) * Ipre * Inim - np.sqrt(3) * Inre * Ipim + Ipim * Inim
    return np.sqrt(valor)


R = 0.01
X = 0.1

Vgp = 0.8
Vgn = 0.1 

Ipre = 0.4
Ipim = 0.0
Inre = 0.4
Inim = 0.0

Vcp = R * Ipre + X * Ipim + np.sqrt(Vgp ** 2 - (X * Ipre - R * Ipim) ** 2)
Vcn = R * Inre - X * Inim + np.sqrt(Vgn ** 2 - (R * Inim + X * Inre) ** 2)
print(Vcp, Vcn)
 
I012 = np.array([0, Ipre - 1j * Ipim, Inre + 1j * Inim])
Iabc = V012_to_abc(I012)
print(abs(Iabc))

Iaf = fIa(Ipre, Ipim, Inre, Inim)
Ibf = fIb(Ipre, Ipim, Inre, Inim)
Icf = fIc(Ipre, Ipim, Inre, Inim)
print(Iaf, Ibf, Icf)