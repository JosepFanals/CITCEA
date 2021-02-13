import numpy as np

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

Va = 1
Vb = 1 * np.exp(-1j * 2 * np.pi / 3)
Vc = 1 * np.exp(1j * 2 * np.pi / 3)
Vabc = np.array([Va, Vb, Vc])
V012 = np.dot(T, Vabc)

Ia = 0.1 * np.exp(1j * -0.1)
Ib = 0.1 * np.exp(1j * (-0.1 - 2 * np.pi / 3))
Ic = 0.1 * np.exp(1j * (-0.1 + 2 * np.pi / 3))
Iabc = np.array([Ia, Ib, Ic])
I012 = np.dot(T, Iabc)

Sa = 1 / 2 * Va * np.conj(Ia)
Sb = 1 / 2 * Vb * np.conj(Ib)
Sc = 1 / 2 * Vc * np.conj(Ic)
print(Sa + Sb + Sc)

print(3 / 2 * np.conj(I012[1]) * V012[1])


