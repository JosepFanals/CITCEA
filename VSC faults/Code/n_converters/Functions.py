import numpy as np


def xabc_to_012(Vabc):
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


def x012_to_abc(V012):
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

    Vabc = np.dot(T, V012)
    return Vabc


def build_static_objects(V_mod, Zv1, Zv2, Zt, Y_con, Y_gnd):
    # Admittances to build the matrices
    Yv1 = 1 / Zv1
    Yv2 = 1 / Zv2
    Yt = 1 / Zt

    # Admittance matrices
    m0 = np.zeros((3,3), dtype=complex)

    Yv1_m = np.zeros((3,3), dtype=complex)
    np.fill_diagonal(Yv1_m, Yv1)

    Yv2_m = np.zeros((3,3), dtype=complex)
    np.fill_diagonal(Yv2_m, Yv2)

    Yt_m = np.zeros((3,3), dtype=complex)
    np.fill_diagonal(Yt_m, Yt)

    Yf_m = Yv1_m + Yv2_m + Yt_m
    Yf_m[0,:] += [Y_gnd[0] + Y_con[0] + Y_con[2], -Y_con[0], -Y_con[2]]
    Yf_m[1,:] += [-Y_con[0], Y_gnd[1] + Y_con[0] + Y_con[1], -Y_con[1]]
    Yf_m[2,:] += [-Y_con[2], -Y_con[1], Y_gnd[2] + Y_con[2] + Y_con[1]]

    m1 = np.block([[Yv1_m, m0, -Yv1_m], [m0, Yv2_m, -Yv2_m], [-Yv1_m, -Yv2_m, Yf_m]])
    m2 = np.block([[m0, m0, m0], [m0, m0, m0], [m0, m0, -Yt_m]])
    m1_inv = np.linalg.inv(m1)

    Vg_v = np.zeros((9,1), dtype=complex)
    a = np.exp(120 * np.pi / 180 * 1j)
    Vg_v[6:9] = [[V_mod], [V_mod * a ** 2], [V_mod * a]]

    return [m1_inv, m2, Vg_v]


def volt_solution(x):
    m1_inv = static_objects[0]
    m2 = static_objects[1]
    Vg_v = static_objects[2] 

    Ii_v = np.zeros((9,1), dtype=complex)
    Ii_v[0:3] = [[x[0]], [x[1]], [x[2]]]
    Ii_v[3:6] = [[x[3]], [x[4]], [x[5]]]

    lhs = Ii_v - np.dot(m2, Vg_v)
    Vv_v = np.dot(m1_inv, lhs)
    return Vv_v


def constraint_Imax(x):
    return Imax - max(abs(x))


def objective_f(x):
    suma = 0
    Vv_v = volt_solution(static_objects, x)
    for kk in range(int(len(V12_vec) / 2)):
        suma += lam_vec[kk] * (abs(1 - abs(V12_vec[kk]))) 
    for kk in range(int(len(V12_vec) / 2), len(V12_vec)):
        suma += lam_vec[kk] * (abs(0 - abs(V12_vec[kk])))
    return suma

