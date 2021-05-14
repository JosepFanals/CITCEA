# from Main import Vp1_vec
import numpy as np
from Functions import xabc_to_012, x012_to_abc, build_static_objects, build_static_objects1 
np.set_printoptions(precision=4)

def fGridCode(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec, Ii_t):

    # Functions
    def volt_solution1(x):
        m1_inv = static_objects[0]
        Ig_v = static_objects[1] 

        Ii_v = np.zeros((6,1), dtype=complex)
        Ii_v[0:3] = [[x[0]], [x[1]], [x[2]]]
        Ii_v[3:6] = [Ig_v[0], Ig_v[1], Ig_v[2]]

        Vv_v = np.dot(m1_inv, Ii_v)
        return Vv_v

    static_objects = build_static_objects1(V_mod, Zv1, Zt, Y_con, Y_gnd)

    # OPTIMIZE
    def f_V1V2(x):
        x = np.asarray(x)
        Vv_v = volt_solution1(x)
        V_p1_abc = Vv_v[0:3]
        V_p1_012 = xabc_to_012(V_p1_abc)

        Vp1 = V_p1_012[1]
        Vn1 = V_p1_012[2]

        # suma = lam_vec[0] * abs(1 - abs(Vp1)) + lam_vec[1] * abs(0 - abs(Vn1)) 
        return [Vp1, Vn1]

    Iabc = [0, 0, 0]
    kpn = 2.5
    fr = 1

    for kk in range(100):  # change for a while
        v1v2 = f_V1V2(Iabc)
        v1 = v1v2[0]
        v2 = v1v2[1]
        
        if abs(v1) < 0.5:
            i1 = fr * 1
        elif abs(v1) < 0.9:
            i1 = fr * kpn * (0.9 - abs(v1))
        else:
            i1 = 0
        
        if abs(v2) > 0.5:
            i2 = fr * 1
        elif abs(v2) > 0.1:
            i2 = fr * kpn * (abs(v2) - 0.1)
        else:
            i2 = 0

        ang1 = np.angle(v1)
        ang2 = np.angle(v2)
        i1 = i1 * np.exp(1j * (ang1 - np.pi / 2))
        i2 = i2 * np.exp(1j * (ang2 + np.pi / 2))
        
        i012 = [0, i1, i2]
        Iabc = x012_to_abc(i012)
        Iabc_max = max(abs(Iabc[0]), abs(Iabc[1]), abs(Iabc[2]))

        if Iabc_max > 1:
            fr = 1 / Iabc_max
        else:
            fr = 1


    return [i1, i2, abs(v1), abs(v2), Iabc]
    # return I_sol

