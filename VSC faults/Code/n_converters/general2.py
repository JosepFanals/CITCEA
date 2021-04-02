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

Zag = 0.2
Yag = 1 / Zag
Ybg = 0
Ycg = 0

Zta = 0.01 + 0.1 * 1j
Ztb = 0.01 + 0.1 * 1j
Ztc = 0.01 + 0.1 * 1j
Yta = 1 / Zta
Ytb = 1 / Ztb
Ytc = 1 / Ztc

mat_A = np.zeros((3,3), dtype=complex)
mat_A[0,0] = Yva
mat_A[1,1] = Yvb
mat_A[2,2] = Yvc

mat_B = - mat_A
mat_C = mat_B

mat_D = np.zeros((3,3), dtype=complex)
mat_D[0,0] = Yab + Yac + Yta + Yag + Yva
mat_D[0,1] = - Yab
mat_D[0,2] = - Yac

mat_D[1,0] = - Yab
mat_D[1,1] = Yab + Ybc + Ytb + Ybg + Yvb
mat_D[1,2] = - Ybc

mat_D[2,0] = - Yac
mat_D[2,1] = - Ybc
mat_D[2,2] = Yac + Ybc + Ytc + Ycg + Yvc

mat_E = np.zeros((3,3), dtype=complex)
mat_F = mat_E
mat_G = mat_E
mat_H = np.zeros((3,3), dtype=complex)
mat_H[0,0] = - Yta
mat_H[1,1] = - Ytb
mat_H[2,2] = - Ytc

mat_1 = np.block([[mat_A, mat_B], [mat_C, mat_D]])
mat_2 = np.block([[mat_E, mat_F], [mat_G, mat_H]])

mat_vg = np.zeros((6,1), dtype=complex)
a = 1 * np.exp(-120 * np.pi / 180 * 1j)
mat_vg[3,0] = 1
mat_vg[4,0] = 1 * a
mat_vg[5,0] = 1 * a ** 2

mat_i = np.zeros((6,1), dtype=complex)
mat_i[0,0] = 0.5
mat_i[1,0] = -0.3 + 0.2 * 1j
mat_i[2,0] = -0.2 - 0.2 * 1j

lhs = mat_i - np.dot(mat_2, mat_vg)
mat_v = np.dot(np.linalg.inv(mat_1), lhs)

V_p = mat_v[0:3]
V_f = mat_v[3:6]

print(V_p)
print(V_f)