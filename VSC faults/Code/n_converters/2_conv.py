import numpy as np
from Functions import xabc_to_012, x012_to_abc

# Connections:
V_mod = 1
Zv1 = 0.01 + 0.05 * 1j
Zv2 = 0.03 + 0.06 * 1j
Zt = 0.01 + 0.1 * 1j

Yv1 = 1 / Zv1
Yv2 = 1 / Zv2
Yt = 1 / Zt

# Faults
Yab = 0
Ybc = 0
Yac = 0

Zag = 0.2
Yag = 1 / Zag
Ybg = 0
Ycg = 0

# Admittance matrices
m0 = np.zeros((3,3), dtype=complex)

Yv1_m = np.zeros((3,3), dtype=complex)
np.fill_diagonal(Yv1_m, Yv1)

Yv2_m = np.zeros((3,3), dtype=complex)
np.fill_diagonal(Yv2_m, Yv2)

Yt_m = np.zeros((3,3), dtype=complex)
np.fill_diagonal(Yt_m, Yt)

Yf_m = Yv1_m + Yv2_m + Yt_m
Yf_m[0,:] += [Yag + Yab + Yac, -Yab, -Yac]
Yf_m[1,:] += [-Yab, Ybg + Yab + Ybc, -Ybc]
Yf_m[2,:] += [-Yac, -Ybc, Ycg + Yac + Ybc]

m1 = np.block([[Yv1_m, m0, -Yv1_m], [m0, Yv2_m, -Yv2_m], [-Yv1_m, -Yv2_m, Yf_m]])
m2 = np.block([[m0, m0, m0], [m0, m0, m0], [m0, m0, -Yt_m]])

Vg_v = np.zeros((9,1), dtype=complex)
a = 1 * np.exp(-120 * np.pi / 180 * 1j)
Vg_v[6:9] = [[V_mod], [V_mod * a ** 2], [V_mod * a]]

Ii_v = np.zeros((9,1), dtype=complex)
Ii_v[0:3] = [[0.5], [-0.3 + 0.2 * 1j], [-0.2 - 0.2 * 1j]]
Ii_v[3:6] = [[0.5], [-0.3 + 0.2 * 1j], [-0.2 - 0.2 * 1j]]

lhs = Ii_v - np.dot(m2, Vg_v)
mat_v = np.dot(np.linalg.inv(m1), lhs)

V_p1 = mat_v[0:3]
V_p2 = mat_v[3:6]
V_f = mat_v[6:9]

print(V_p1)
print(V_p2)
print(V_f)