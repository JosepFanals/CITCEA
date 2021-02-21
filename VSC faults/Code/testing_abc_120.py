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

Va = 1 / (Z2 + Zf) * (Ia * (Z1 * Z2 + Z1 * Zf + Z2 * Zf) + Vga * Zf)
Vb = Vgb + Ib * (Z1 + Z2)
Vc = Vgc + Ic * (Z1 + Z2)
print(Va, Vb, Vc)
Vlg_abc = np.array([Va, Vb, Vc])
Vlg_120 = Vabc_to_012(Vlg_abc)
print(Vlg_120)

# Vc1 = 1 / (3 * Zf + 3 * Z2) * (1 * (3 * Zf + 2 * Z2) + I1 * (3 * Z1 * Zf + 3 * Z1 * Z2 + 3 * Z2 * Zf + 2 * Z2 ** 2) + I2 * (-Z2 ** 2))
# Vc1 = 1 / (3 * Zf + 3 * Z2) * (1 * (3 * Zf + 2 * Z2) + I1 * (3 * Z2 * Z1 + 3 * Zf * Z1 + 3 * Zf * Z2 + 2 * Z2 * Z2) + I2 * (-5 * Z2 * Z2 - 6 * Zf * Z2))
Vc1 = I1 * (Z1 + Z2) + 1 - Z2 / (3 * Zf + 3 * Z2) * (I2 * Z2 + I1 * Z2 + 1)
Vc2 = I2 * (Z1 + Z2) - Z2 / (3 * Zf + 3 * Z2) * (I2 * Z2 + I1 * Z2 + 1)
Vc0 = - Z2 / (3 * Zf + 3 * Z2) * (1 + I1 * Z2 + I2 * Z2)
print(Vc0, Vc1, Vc2)
Vcc012 = np.array([Vc0, Vc1, Vc2])
Vccabc = V012_to_abc(Vcc012)
print(Vccabc)


print('=========')

# LL FAULT

Va = Ia * (Z1 + Z2) + Vga
# Vb = Ib * Z1 + 1 / (3 * Z2 + Zf) * (Ic * (Z2 * Z2) + Ib * (Z2 * Zf + Z2 * Z2) + Vgc * Z2 + Vgb * (Z2 + Zf))
# Vc = Ic * Z1 + (Zf + Z2) / Z2 * 1 / (3 * Z2 + Zf) * (Ic * (Z2 * Z2) + Ib * (Z2 * Zf + Z2 * Z2) + Vgc * Z2 + Vgb * (Z2 * Zf)) - Vgb * Zf / Z2 - Ib * Z2 * Zf / Z2
# Vb = Vgb + Ia * (-2 * Z2) + Ic * Z2 + Z2 / (2 * Z2 + Zf) * (Vgc - Vgb - Zf * Ic + Z2 * Ia)
# Vc = Ic * (2 * Z1 + Zf) + Ia * Z1 + Vgb + Ia * (-2 * Z2) + Ic * Z2 + Z2 / (2 * Z2 + Zf) * (Vgc - Vgb - Zf * Ic + Z2 * Ia) + Zf / (2 * Z2 + Zf) * (Vgc - Vgb - Zf * Ic + Z2 * Ia)

# Vy = 1 / (Zf + Z2 - Z2 * Z2 / (Zf + Z2)) * (Z2 * Zf * Ic + Zf * Vgc + Z2 / (Zf + Z2) * (Z2 * Zf * Ib + Zf * Vgb))
# Vy = (Zf + Z2) / (Zf * Zf + 2 * Zf * Z2) * (Z2 * Zf * Ic + Zf * Vgc + Z2 / (Zf + Z2) * (Z2 * Zf * Ib + Zf * Vgb))
Vy = 1 / (Zf + 2 * Z2) * (Ic * (Z2 * (Zf + Z2)) + Vgc * (Zf + Z2) + Ib * Z2 * Z2 + Vgb * Z2)
#Vx = 1 / (Zf + Z2) * (Z2 * Zf * Ib + Zf * Vgb + Z2 * Vy)
Vx = 1 / ((Zf + Z2) * (Zf + 2 * Z2)) * (Ib * (Z2 * Zf * Zf + 2 * Z2 * Z2 * Zf + Z2 * Z2 * Z2) + Ic * (Z2 * Z2 * Zf + Z2 * Z2 * Z2) + Vgb * (Zf * Zf + 2 * Zf * Z2 + Z2 * Z2) + Vgc * (Z2 * Zf + Z2 * Z2))

# Vy = 1 / (Zf + 2 * Z2) * (Ic * (Z2 * Zf + Z2 * Z2) + Ib * (Z2 * Zf) + Vgc * (Zf + Z2) + Vgb * Z2)
# Vx = 1 / (Zf * Zf + 3 * Z2 * Zf + 2 * Z2 * Z2) * (Ib * (Z2 * Zf * Zf + 3 * Z2 * Z2 * Zf) + Vgb * (Zf * Zf + 2 * Zf * Z2 + Z2 * Z2) + Ic * (Z2 * Z2 * Zf + Z2 * Z2 * Z2) + Vgc * (Z2 * Zf + Z2 * Z2))

Vc = Vy + Ic * Z1
Vb = Vx + Ib * Z1

# print(Vy, Vx)

print(Va, Vb, Vc)


# Vc2 = 1 / (2 * Z2 + Zf) * (1 + I1 * (Z1 + Z2) + I2 * (Z1 + Z2) * Zf / Z2)
# Vc1 = 1 + I1 * (Z1 + Z2) + I2 * (Z1 + Z2) - Vc2
# Vc1 = 1 / (Zf * Zf + 3 * Z2 * Zf + 2 * Z2 * Z2) * (1 * (Zf * Zf + Z2 * Z2 + 2 * Z2 * Zf) + I1 * (2 * Z1 * Z2 * Z2 + 3 * Z1 * Z2 * Zf + 2 * Z2 * Z2 * Zf + 2 * Z1 * Z2 * Z2 + Z2 * Z2 * Z2) + I2 * (Z2 * Z2 * Zf + Z2 * Z2 * Z2))
# Vc2 = 1 / (Zf + 2 * Z2 * Zf) * (1 * Z2 * Zf + I1 * (Z2 * Z2 * Zf) + I2 * (Z1 * Zf * Zf + Z2 * Zf * Zf + 2 * Z1 * Z2 * Zf + Z2 * Z2 * Zf))

#Ix = 1 / (2 * Z2 + Zf) * (1 + I1 * Z2 - I2 * Z2)
#Vc2 = 1 + I1 * Z2 + I2 * Z1 - Ix * (Z2 + Zf)
#Vc1 = Z1 * I1 + Z2 * (I1 - Ix) + 1

Vc1 = 1 + I1 * Z1 + I1 * Z2 - Z2 / (2 * Z2 + Zf) * (1 + I1 * Z2 - I2 * Z2)
Vc2 = 1 + I1 * Z2 + I2 * Z1 - (Z2 + Zf) / (2 * Z2 + Zf) * (1 + I1 * Z2 - I2 * Z2)

Vc0 = 0
Vccc012 = np.array([Vc0, Vc1, Vc2])
Vcccabc = V012_to_abc(Vccc012)
print(Vcccabc)


print('----------')

# LLG FAULT
Vc1 = I1 * Z1 + (Z2 + 3 * Zf) / (3 * Z2 + 6 * Zf) * (I1 * Z2 + I2 * Z2 + 1)
Vc2 = I2 * Z1 + (Z2 + 3 * Zf) / (3 * Z2 + 6 * Zf) * (I1 * Z2 + I2 * Z2 + 1)
Vc0 = Z2 / (3 * Z2 + 6 * Zf) * (I1 * Z2 + I2 * Z2 + 1)
Vc012 = np.array([Vc0, Vc1, Vc2])
Vcabc = V012_to_abc(Vc012)
print(Vcabc)

Vca = Ia * (Z1 + Z2) + Vga
Vcb = Z1 * Ib + (Z2 * Zf * (Ib + Ic) + Zf * (Vgb + Vgc)) / (2 * Zf + Z2)
Vcc = Z1 * Ic + (Z2 * Zf * (Ib + Ic) + Zf * (Vgb + Vgc)) / (2 * Zf + Z2)
print(Vca, Vcb, Vcc)


