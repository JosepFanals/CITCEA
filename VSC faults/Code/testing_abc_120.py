import numpy as np


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

    # Tinv = np.linalg.inv(T)
    Vabc = np.dot(T, V012)
    return Vabc


Zf = 0.1
Z2 = 0.05
Z1 = 0.08

Vga = 1
alph = np.exp(1j * -120 * np.pi / 180)
Vgb = 1 * alph
Vgc = 1 * alph ** 2

Ia = 1
Ib = -0.5
Ic = -0.5

# THREE PHASE FAULT
Va = 1 / (Zf + Z2) * (Ia * (Z1 * Z2 + Z2 * Zf + Zf * Z1) + Vga * Zf)
Vb = 1 / (Zf + Z2) * (Ib * (Z1 * Z2 + Z2 * Zf + Zf * Z1) + Vgb * Zf)
Vc = 1 / (Zf + Z2) * (Ic * (Z1 * Z1 + Z2 * Zf + Zf * Z1) + Vgc * Zf)

print(Va, Vb, Vc)

Iabc = np.array([Ia, Ib, Ic])
I012 = Vabc_to_012(Iabc)
I0 = I012[0]
I1 = I012[1]
I2 = I012[2]

Vc0 = 0
Vc1 = 1 / (Zf + Z2) * (1 * Zf + I1 * (Z1 * Zf + Z1 * Z2 + Zf * Z2))
Vc2 = 1 / (Zf + Z2) * (I2 * (Z2 * Zf + Z1 * Z2 + Z1 * Zf))
Vc012 = np.array([Vc0, Vc1, Vc2])
Vcabc = V012_to_abc(Vc012)
print(Vcabc)

print('=========')

# LG FAULT
Va = Vga + Ia * (Z1 + Z2)
Vb = Vgb + Ib * (Z1 + Z2)
Vc = 1 / (1/Z2 + 1/Zf) * (Ic + Vgc / Z2)
print(Va, Vb, Vc)
