# from Main import Vp1_vec
import numpy as np

from mystic.symbolic import generate_constraint, generate_solvers, simplify
from mystic.symbolic import generate_penalty, generate_conditions
from mystic.solvers import fmin, fmin_powell, diffev, diffev2
from mystic.penalty import quadratic_equality, quadratic_inequality
# from mystic.symbolic import absval

from scipy.optimize import minimize
from scipy.optimize.optimize import _minimize_scalar_bounded
from Functions import xabc_to_012, x012_to_abc, build_static_objects 
np.set_printoptions(precision=4)

def fOptimal_mystic(V_mod, Imax, Zv1, Zv2, Zt, Y_con, Y_gnd, lam_vec, Ii_t):

    # Functions
    def volt_solution(x):
        m1_inv = static_objects[0]
        Ig_v = static_objects[1] 

        Ii_v = np.zeros((9,1), dtype=complex)
        Ii_v[0:3] = [[x[0] + 1j * x[1]], [x[2] + 1j * x[3]], [x[4] + 1j * x[5]]]
        Ii_v[3:6] = [[x[6] + 1j * x[7]], [x[8] + 1j * x[9]], [x[10] + 1j * x[11]]]
        Ii_v[6:9] = [Ig_v[0], Ig_v[1], Ig_v[2]]

        Vv_v = np.dot(m1_inv, Ii_v)
        return Vv_v

    def constraint_Imax(x):
        # I1a = abs(x[0] + 1j * x[1])
        # I1b = abs(x[2] + 1j * x[3])
        # I1c = abs(x[4] + 1j * x[5])
        # I2a = abs(x[6] + 1j * x[7])
        # I2b = abs(x[8] + 1j * x[9])
        # I2c = abs(x[10] + 1j * x[11])

        I1a2 = x[0] ** 2 + x[1] ** 2
        I1b2 = x[2] ** 2 + x[3] ** 2
        I1c2 = x[4] ** 2 + x[5] ** 2
        I2a2 = x[6] ** 2 + x[7] ** 2
        I2b2 = x[8] ** 2 + x[9] ** 2
        I2c2 = x[10] ** 2 + x[11] ** 2
        return Imax ** 2 - max(I1a2, I1b2, I1c2, I2a2, I2b2, I2c2)

    def constraint_Ire1(x):
        return x[0] + x[2] + x[4] 

    def constraint_Iim1(x):
        return x[1] + x[3] + x[5] 

    def constraint_Ire2(x):
        return x[6] + x[8] + x[10] 

    def constraint_Iim2(x):
        return x[7] + x[9] + x[11] 



    # Data
    # V_mod = 1
    # Imax = 1
    # Zv1 = 0.01 + 0.05 * 1j
    # Zv2 = 0.03 + 0.06 * 1j
    # Zt = 0.01 + 0.1 * 1j
    # Y_con = [0, 0, 0]  # Yab, Ybc, Yac
    # Y_gnd = [15, 0, 0]  # Yag, Ybg, Yc
    # lam_vec = [1, 1, 1, 1]  # V1p, V2p, V1n, V2n
    # Ii_t = [0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]

    static_objects = build_static_objects(V_mod, Zv1, Zv2, Zt, Y_con, Y_gnd)

    # Optimize
    con1 = {'type': 'ineq', 'fun': constraint_Imax}
    con2 = {'type': 'eq', 'fun': constraint_Ire1}
    con3 = {'type': 'eq', 'fun': constraint_Iim1}
    con4 = {'type': 'eq', 'fun': constraint_Ire2}
    con5 = {'type': 'eq', 'fun': constraint_Iim2}
    cons = [con1, con2, con3, con4, con5]

    # @quadratic_inequality(con1, k=1e4)
    def objective_f(x):
        Vv_v = volt_solution(x)
        V_p1_abc = Vv_v[0:3]
        V_p2_abc = Vv_v[3:6]
        V_p1_012 = xabc_to_012(V_p1_abc)
        V_p2_012 = xabc_to_012(V_p2_abc)

        Vp1 = V_p1_012[1]
        Vp2 = V_p2_012[1]
        Vn1 = V_p1_012[2]
        Vn2 = V_p2_012[2]

        # suma = lam_vec[0] * abs(1 - abs(V_p1_012[1])) + lam_vec[1] * abs(1 - abs(V_p2_012[1])) + lam_vec[2] * abs(0 - abs(V_p1_012[2])) + lam_vec[3] * abs(0 - abs(V_p2_012[2]))
        # suma = lam_vec[0] * (1 - np.real(Vp1 * np.conj(Vp1))) + lam_vec[1] * (1 - np.real(Vp2 * np.conj(Vp2))) + lam_vec[2] * np.real(Vn1 * np.conj(Vn1)) + lam_vec[3] * np.real(Vn2 * np.conj(Vn2))
        # suma = (1 - Vp1 * np.conj(Vp1)) ** 2 + (1 - Vp2 * np.conj(Vp2)) ** 2 
        # suma = np.real((1 - Vp1 * np.conj(Vp1)) ** 2 + (0 + Vn1 * np.conj(Vn1)) ** 2 + (1 - Vp2 * np.conj(Vp2)) ** 2 + (0 + Vn2 * np.conj(Vn2)) ** 2)
        suma = np.real((1 - Vp1 * np.conj(Vp1)) ** 2 + (0 + Vn1 * np.conj(Vn1)) ** 2)
        return suma


    def obj_fun(x):
        Vv_v = volt_solution(x)
        V_p1_abc = Vv_v[0:3]
        V_p2_abc = Vv_v[3:6]
        V_p1_012 = xabc_to_012(V_p1_abc)
        V_p2_012 = xabc_to_012(V_p2_abc)

        Vp1 = V_p1_012[1]
        Vp2 = V_p2_012[1]
        Vn1 = V_p1_012[2]
        Vn2 = V_p2_012[2]

        # suma = np.real((1 - Vp1 * np.conj(Vp1)) ** 2 + (0 + Vn1 * np.conj(Vn1)) ** 2)
        # suma = np.real((1 - Vp1 * np.conj(Vp1)) ** 2 + (0 + Vn1 * np.conj(Vn1)) ** 2 + (1 - Vp2 * np.conj(Vp2)) ** 2 + (0 + Vn2 * np.conj(Vn2)) ** 2)
        # suma = np.real(lam_vec[0] * (1 - Vp1 * np.conj(Vp1)) ** 2 + lam_vec[1] * (0 + Vn1 * np.conj(Vn1)) ** 2 + lam_vec[2] * (1 - Vp2 * np.conj(Vp2)) ** 2 + lam_vec[3] * (0 + Vn2 * np.conj(Vn2)) ** 2)
        # suma = lam_vec[0] * (1 - abs(Vp1)) ** 2 + lam_vec[1] * (0 + abs(Vn1)) ** 2 + lam_vec[2] * (1 - abs(Vp2)) ** 2 + lam_vec[3] * (0 + abs(Vn2)) ** 2
        suma = lam_vec[0] * (1 - abs(Vp1)) + lam_vec[1] * (0 + abs(Vn1)) + lam_vec[2] * (1 - abs(Vp2)) + lam_vec[3] * (0 + abs(Vn2))
        # suma = lam_vec[0] * (1 - abs(Vp1)) ** 2 + lam_vec[1] * (0 + abs(Vn1)) ** 2 
        return suma

    def penalty_mean(x, target):
        a1 = x[0] ** 2 + x[1] ** 2
        a2 = x[2] ** 2 + x[3] ** 2
        a3 = x[4] ** 2 + x[5] ** 2
        a4 = x[6] ** 2 + x[7] ** 2
        a5 = x[8] ** 2 + x[9] ** 2
        a6 = x[10] ** 2 + x[11] ** 2
        return (max(a1, a2, a3, a4, a5, a6) - target)

    # @quadratic_equality(condition=penalty_mean, kwds={'target':1.0})
    @quadratic_inequality(condition=penalty_mean, kwds={'target':1.0})
    def penalty(x):
        return 0.0


    equations_c = """
    x0 + x2 + x4 == 0
    x1 + x3 + x5 == 0
    x6 + x8 + x10 == 0
    x7 + x9 + x11 == 0
    """

    cf = generate_constraint(generate_solvers(simplify(equations_c)))

    bnds = [(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),(-Imax, Imax),]
    # sol = minimize(objective_f, Ii_t, method='SLSQP', constraints=cons, bounds=bnds, options={'ftol':1e-12, 'maxiter':10000})


    # result = fmin(obj_fun, x0=Ii_t, bounds=bnds, penalty=penalty, constraints=cf, npop=10000, gtol=10000, disp=True, full_output=True, ftol=1e-14, maxiter=15000, maxfun=15000)
    # result = fmin(obj_fun, x0=Ii_t, bounds=bnds, penalty=penalty, constraints=cf, npop=10, gtol=10, disp=True, full_output=True, ftol=1e-6, maxiter=35000, maxfun=35000)
    # result = fmin(obj_fun, x0=Ii_t, bounds=bnds, constraints=cf, npop=100, gtol=100, disp=True, full_output=True, ftol=1e-5)
    # result = fmin_powell(obj_fun, x0=Ii_t, bounds=bnds, constraints=cf, npop=100, gtol=100, disp=False, ftol=1e-13)
    # result = fmin_powell(cost=obj_fun, x0=Ii_t, bounds=bnds, penalty=penalty, constraints=cf, npop=100, gtol=100, disp=True, full_output=True, ftol=1e-15, maxiter=1750000, maxfun=1750000, scale=0.5, cross=0.5)
    result = diffev(cost=obj_fun, x0=Ii_t, bounds=bnds, penalty=penalty, constraints=cf, npop=10, gtol=10, disp=True, full_output=True, ftol=1e-5, maxiter=1750000, maxfun=1750000, scale=0.5, cross=0.5)

    # result = diffev2(objective_f, x0=Ii_t, npop=10, gtol=200, disp=False, full_output=True, itermon=mon, maxiter=1000)

    I_sol = result
    # ff_obj = obj_fun(I_sol)
    # I_sol = sol.x
    # print(I_sol)

    # Manage results
    # print(I_sol[1])
    # I1_abc = [I_sol[0] + 1j * I_sol[1], I_sol[2] + 1j * I_sol[3], I_sol[4] + 1j * I_sol[5]]
    # I2_abc = [I_sol[6] + 1j * I_sol[7], I_sol[8] + 1j * I_sol[9], I_sol[10] + 1j * I_sol[11]]
    # V_f = volt_solution(I_sol)
    # V_p1_012 = xabc_to_012(V_f[0:3])
    # V_p2_012 = xabc_to_012(V_f[3:6])
    # Ip1_012 = xabc_to_012(I1_abc)
    # Ip2_012 = xabc_to_012(I2_abc)
    # Ip1_1 = Ip1_012[1] * np.exp(-1j * np.angle(V_p1_012[1]))
    # Ip1_2 = Ip1_012[2] * np.exp(-1j * np.angle(V_p1_012[2]))
    # Ip2_1 = Ip2_012[1] * np.exp(-1j * np.angle(V_p2_012[1]))
    # Ip2_2 = Ip2_012[2] * np.exp(-1j * np.angle(V_p2_012[2]))
    # print(sol.fun)
    # print(sol.success)

    # print(I_sol[0])

    I1_abc = [I_sol[0][0] + 1j * I_sol[0][1], I_sol[0][2] + 1j * I_sol[0][3], I_sol[0][4] + 1j * I_sol[0][5]]
    I2_abc = [I_sol[0][6] + 1j * I_sol[0][7], I_sol[0][8] + 1j * I_sol[0][9], I_sol[0][10] + 1j * I_sol[0][11]]
    V_f = volt_solution(I_sol[0])
    V_p1_012 = xabc_to_012(V_f[0:3])
    V_p2_012 = xabc_to_012(V_f[3:6])
    Ip1_012 = xabc_to_012(I1_abc)
    Ip2_012 = xabc_to_012(I2_abc)
    Ip1_1 = Ip1_012[1] * np.exp(-1j * np.angle(V_p1_012[1]))
    Ip1_2 = Ip1_012[2] * np.exp(-1j * np.angle(V_p1_012[2]))
    Ip2_1 = Ip2_012[1] * np.exp(-1j * np.angle(V_p2_012[1]))
    Ip2_2 = Ip2_012[2] * np.exp(-1j * np.angle(V_p2_012[2]))
    # print(sol.fun)
    # print(sol.success)

    # print(I_sol)

    return [Ip1_1, Ip1_2, Ip2_1, Ip2_2, abs(V_p1_012[1]), abs(V_p1_012[2]), abs(V_p2_012[1]), abs(V_p2_012[2]), I_sol]
    # return I_sol
