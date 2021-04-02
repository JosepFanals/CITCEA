import numpy as np

Zva = 0.01 + 0.05 * 1j
Zvb = 0.01 + 0.05 * 1j
Zvc = 0.01 + 0.05 * 1j
Yva = 1 / Zva
Yvb = 1 / Zvb
Yvc = 1 / Zvc

Yab = 0
Ybc = 0
Yac = 0

Yag = 0
Ybg = 0
Ycg = 0

Zta = 0.01 + 0.1 * 1j
Ztb = 0.01 + 0.1 * 1j
Ztc = 0.01 + 0.1 * 1j
Yta = 1 / Zta
Ytb = 1 / Ztb
Ytc = 1 / Ztc

mat_1 = np.zeros((3,3), dtype=complex)
mat_1[0,0] = Yab + Yac + Yta + Yag
mat_1[0,1] = - Yab
mat_1[0,2] = - Yac

mat_1[1,0] = - Yab
mat_1[1,1] = Yab + Ybc + Ytb + Ybg
mat_1[1,2] = - Ybc

mat_1[2,0] = - Yac
mat_1[2,1] = - Ybc
mat_1[2,2] = Yac + Ybc + Ytc + Ycg

mat_2 = np.zeros((3,3), dtype=complex)
mat_2[0,0] = - Yta
mat_2[1,1] = - Ytb
mat_2[2,2] = - Ytc

mat_vg = np.zeros((3,1), dtype=complex)
a = 1 * np.exp(-120 * np.pi / 180 * 1j)
mat_vg[0,0] = 1
mat_vg[1,0] = 1 * a
mat_vg[2,0] = 1 * a ** 2

mat_i = np.zeros((3,1), dtype=complex)
mat_i[0,0] = 0.5
mat_i[1,0] = -0.3 + 0.2 * 1j
mat_i[2,0] = -0.2 - 0.2 * 1j

lhs = mat_i - np.dot(mat_2, mat_vg)
mat_vf = np.dot(np.linalg.inv(mat_1), lhs)
print(mat_vf)