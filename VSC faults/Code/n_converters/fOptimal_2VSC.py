import numpy as np
from scipy.optimize import minimize
from Functions import xabc_to_012, x012_to_abc, build_static_objects 
np.set_printoptions(precision=4)

def fOptimal(V_mod, Imax, Zv1, Zv2, Zt, Y_con, Y_gnd, lam_vec, nnn):

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
        I1a = abs(x[0] + 1j * x[1])
        I1b = abs(x[2] + 1j * x[3])
        I1c = abs(x[4] + 1j * x[5])
        I2a = abs(x[6] + 1j * x[7])
        I2b = abs(x[8] + 1j * x[9])
        I2c = abs(x[10] + 1j * x[11])
        return Imax - max(I1a, I1b, I1c, I2a, I2b, I2c)

    def constraint_Ire1(x):
        return x[0] + x[2] + x[4] 

    def constraint_Iim1(x):
        return x[1] + x[3] + x[5] 

    def constraint_Ire2(x):
        return x[6] + x[8] + x[10] 

    def constraint_Iim2(x):
        return x[7] + x[9] + x[11] 

    def objective_f(x):
        Vv_v = volt_solution(x)
        V_p1_abc = Vv_v[0:3]
        V_p2_abc = Vv_v[3:6]
        V_p1_012 = xabc_to_012(V_p1_abc)
        V_p2_012 = xabc_to_012(V_p2_abc)

        suma = lam_vec[0] * abs(1 - abs(V_p1_012[1])) + lam_vec[1] * abs(1 - abs(V_p2_012[1])) + lam_vec[2] * abs(0 - abs(V_p1_012[2])) + lam_vec[3] * abs(0 - abs(V_p2_012[2]))
        return suma

    # Data
    # V_mod = 1
    # Imax = 1
    # Zv1 = 0.01 + 0.05 * 1j
    # Zv2 = 0.03 + 0.06 * 1j
    # Zt = 0.01 + 0.1 * 1j
    # Y_con = [0, 0, 0]  # Yab, Ybc, Yac
    # Y_gnd = [15, 0, 0]  # Yag, Ybg, Yc
    # lam_vec = [1, 1, 1, 1]  # V1p, V2p, V1n, V2n
    Ii_t = [0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]

    static_objects = build_static_objects(V_mod, Zv1, Zv2, Zt, Y_con, Y_gnd)

    # Optimize
    con1 = {'type': 'ineq', 'fun': constraint_Imax}
    con2 = {'type': 'eq', 'fun': constraint_Ire1}
    con3 = {'type': 'eq', 'fun': constraint_Iim1}
    con4 = {'type': 'eq', 'fun': constraint_Ire2}
    con5 = {'type': 'eq', 'fun': constraint_Iim2}
    cons = [con1, con2, con3, con4, con5]

    sol = minimize(objective_f, Ii_t, method='SLSQP', constraints=cons, options={'ftol':1e-10})
    I_sol = sol.x

    # Manage results
    I1_abc = [I_sol[0] + 1j * I_sol[1], I_sol[2] + 1j * I_sol[3], I_sol[4] + 1j * I_sol[5]]
    I2_abc = [I_sol[6] + 1j * I_sol[7], I_sol[8] + 1j * I_sol[9], I_sol[10] + 1j * I_sol[11]]
    V_f = volt_solution(I_sol)
    V_p1_012 = xabc_to_012(V_f[0:3])
    V_p2_012 = xabc_to_012(V_f[3:6])
    Ip1_012 = xabc_to_012(I1_abc)
    Ip2_012 = xabc_to_012(I2_abc)
    Ip1_1 = Ip1_012[1] * np.exp(-1j * np.angle(V_p1_012[1]))
    Ip1_2 = Ip1_012[2] * np.exp(-1j * np.angle(V_p1_012[2]))
    Ip2_1 = Ip2_012[1] * np.exp(-1j * np.angle(V_p2_012[1]))
    Ip2_2 = Ip2_012[2] * np.exp(-1j * np.angle(V_p2_012[2]))
    print(sol.fun)
    print(sol.success)

    return [Ip1_1, Ip1_2, Ip2_1, Ip2_2, abs(V_p1_012[1]), abs(V_p1_012[2]), abs(V_p2_012[1]), abs(V_p2_012[2]), sol.fun]
