# from Main import Vp1_vec
import numpy as np
import numpy as np
import mystic.symbolic as ms
import mystic.solvers as my
import mystic.math as mm
from mystic.penalty import quadratic_equality, lagrange_equality, linear_equality, uniform_equality, uniform_inequality, quadratic_inequality
from Functions import xabc_to_012, x012_to_abc, build_static_objects, build_static_objects1 
np.set_printoptions(precision=4)

def fROptimal_mystic(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec, Ii_t):

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




    def penalty_A(x):
        x = np.asarray(x)
        Vv_v = volt_solution(x)
        V_p1_abc = Vv_v[0:3]
        return np.imag(V_p1_abc[0] * np.conj(x[0] + 1j * x[1]))

    def penalty_B(x):
        x = np.asarray(x)
        Vv_v = volt_solution(x)
        V_p1_abc = Vv_v[0:3]
        return np.imag(V_p1_abc[1] * np.conj(x[2] + 1j * x[3]))

    def penalty_C(x):
        x = np.asarray(x)
        Vv_v = volt_solution(x)
        V_p1_abc = Vv_v[0:3]
        return np.imag(V_p1_abc[2] * np.conj(x[4] + 1j * x[5]))

    def suma_re(x):
        return x[0] + x[2] + x[4]

    def suma_im(x):
        return x[1] + x[3] + x[5]

    def ia_max(x):
        return x[0]*x[0] + x[1]*x[1] - 1
    
    def ib_max(x):
        return x[2]*x[2] + x[3]*x[3] - 1

    def ic_max(x):
        return x[4]*x[4] + x[5]*x[5] - 1


    # see: https://stackoverflow.com/questions/51892741/constrained-global-optimization-tuning-mystic

    @quadratic_inequality(ia_max)  # vary k=1e12 accordingly
    @quadratic_inequality(ib_max)  # vary k=1e12 accordingly
    @quadratic_inequality(ic_max)  # vary k=1e12 accordingly
    @quadratic_equality(suma_re)
    @quadratic_equality(suma_im)
    @quadratic_equality(penalty_A)
    @quadratic_equality(penalty_B)
    @quadratic_equality(penalty_C)
    def penalty(x):
        return 0.0


    bnds = [(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax)]

    # sol = my.diffev(obj_fun, Ii_t, penalty=pens, disp=True, bounds=bnds, gtol=10, ftol=1e-5, full_output=True, maxiter=100000, maxfun=100000)
    sol = my.diffev(obj_fun, Ii_t, penalty=penalty, disp=True, bounds=bnds, gtol=100, ftol=1e-50, full_output=True, maxiter=100000, maxfun=100000)

    I_sol = sol

    I1_abc = [I_sol[0][0] + 1j * I_sol[0][1], I_sol[0][2] + 1j * I_sol[0][3], I_sol[0][4] + 1j * I_sol[0][5]]
    V_f = volt_solution(I_sol[0])
    V_p1_012 = xabc_to_012(V_f[0:3])
    Ip1_012 = xabc_to_012(I1_abc)
    Ip1_1 = Ip1_012[1] * np.exp(-1j * np.angle(V_p1_012[1]))
    Ip1_2 = Ip1_012[2] * np.exp(-1j * np.angle(V_p1_012[2]))

    return [Ip1_1, Ip1_2, abs(V_p1_012[1]), abs(V_p1_012[2]), I_sol]
    # return I_sol
