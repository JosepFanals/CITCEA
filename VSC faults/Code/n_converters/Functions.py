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

    It_mod = V_mod * Yt

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
    m1_inv = np.linalg.inv(m1)

    Ig_v = np.zeros((3,1), dtype=complex)
    a = np.exp(120 * np.pi / 180 * 1j)
    Ig_v[0:3] = [[It_mod], [It_mod * a ** 2], [It_mod * a]]

    return [m1_inv, Ig_v]


