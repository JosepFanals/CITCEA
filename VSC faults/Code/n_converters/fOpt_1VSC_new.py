# from Main import Vp1_vec
import numpy as np
import numpy as np
import mystic.symbolic as ms
import mystic.solvers as my
import mystic.math as mm
from Functions import xabc_to_012, x012_to_abc, build_static_objects, build_static_objects1 
np.set_printoptions(precision=4)

def fOptimal_mystic(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec, Ii_t):

    # Functions
    def volt_solution(x):
        m1_inv = static_objects[0]
        Ig_v = static_objects[1] 

        Ii_v = np.zeros((6,1), dtype=complex)
        Ii_v[0:3] = [[x[0] + 1j * x[1]], [x[2] + 1j * x[3]], [x[4] + 1j * x[5]]]
        Ii_v[3:6] = [Ig_v[0], Ig_v[1], Ig_v[2]]

        Vv_v = np.dot(m1_inv, Ii_v)
        return Vv_v

    static_objects = build_static_objects1(V_mod, Zv1, Zt, Y_con, Y_gnd)

    # OPTIMIZE
    def obj_fun(x):
        x = np.asarray(x)
        Vv_v = volt_solution(x)
        V_p1_abc = Vv_v[0:3]
        V_p1_012 = xabc_to_012(V_p1_abc)

        Vp1 = V_p1_012[1]
        Vn1 = V_p1_012[2]

        suma = lam_vec[0] * abs(1 - abs(Vp1)) + lam_vec[1] * abs(0 - abs(Vn1)) 
        return suma


    equations_p = """
    x0*x0 + x1*x1 -1 <= 0
    x2*x2 + x3*x3 -1 <= 0
    x4*x4 + x5*x5 -1 <= 0
    x0 + x2 + x4 == 0
    x1 + x3 + x5 == 0
    """

    pens = ms.generate_penalty(ms.generate_conditions(equations_p))
    bnds = [(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax)]

    sol = my.diffev(obj_fun, Ii_t, penalty=pens, disp=True, bounds=bnds, gtol=4, ftol=1e-4, full_output=True, maxiter=100000, maxfun=100000)

    I_sol = sol

    I1_abc = [I_sol[0][0] + 1j * I_sol[0][1], I_sol[0][2] + 1j * I_sol[0][3], I_sol[0][4] + 1j * I_sol[0][5]]
    V_f = volt_solution(I_sol[0])
    V_p1_012 = xabc_to_012(V_f[0:3])
    Ip1_012 = xabc_to_012(I1_abc)
    Ip1_1 = Ip1_012[1] * np.exp(-1j * np.angle(V_p1_012[1]))
    Ip1_2 = Ip1_012[2] * np.exp(-1j * np.angle(V_p1_012[2]))

    return [Ip1_1, Ip1_2, abs(V_p1_012[1]), abs(V_p1_012[2]), I_sol]
    # return I_sol
