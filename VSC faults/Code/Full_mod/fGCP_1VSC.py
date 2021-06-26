import numpy as np
from Functions_pn import Inmax, CurrentP_data
from Functions_main import xabc_to_012, x012_to_abc, build_static_objects, build_static_objects1 
np.set_printoptions(precision=4)


def fGCP_1vsc(V_mod, Imax, Zv1, Zt, Y_con, Y_gnd, lam_vec):

	# Functions
	def volt_solution(x):
        	m1_inv = static_objects[0]
        	Ig_v = static_objects[1] 

        	Ii_v = np.zeros((6,1), dtype=complex)
        	# Ii_v[0:3] = [[x[0] + 1j * x[1]], [x[2] + 1j * x[3]], [x[4] + 1j * x[5]]]
        	Ii_v[0:3] = [[x[0]], [x[1]], [x[2]]]
        	Ii_v[3:6] = [Ig_v[0], Ig_v[1], Ig_v[2]]

        	Vv_v = np.dot(m1_inv, Ii_v)
        	return Vv_v

	static_objects = build_static_objects1(V_mod, Zv1, Zt, Y_con, Y_gnd)

	Iconv_abc = [0, 0, 0]
	Iconv_abc_prev = [1, 1, 1]
	tol = 1e-3

	# loop
	while abs(Iconv_abc[0] - Iconv_abc_prev[0]) > tol or abs(Iconv_abc[1] - Iconv_abc_prev[1]) > tol or abs(Iconv_abc[2] - Iconv_abc_prev[2]) > tol:
		Iconv_abc_prev = Iconv_abc
		Vv_v = volt_solution(Iconv_abc)
		V_p1_abc = Vv_v[0:3]
		V_p1_012 = xabc_to_012(V_p1_abc)

		Vp = V_p1_012[1]
		Vn = V_p1_012[2]

		# Input parameters
		Imax = 1
		Vpabs = np.abs(Vp)
		Vphigh = 0.9
		Vplow = 0.4
		kp = 2
		margin = 0.01 


		# Grid code positive sequence
		Ip = 0
		if Vpabs < Vphigh and Vpabs > Vplow:
			Ip = kp * (Vphigh - Vpabs)
		elif Vpabs <= Vplow:
			Ip = Imax

		Ia_p, ta, ang_na = CurrentP_data(Vp, Ip, Vn, 'a')
		Ib_p, tb, ang_nb = CurrentP_data(Vp, Ip, Vn, 'b')
		Ic_p, tc, ang_nc = CurrentP_data(Vp, Ip, Vn, 'c')


		# Calculation of max negative sequence current:
		Iann = Inmax(Ia_p, ta, Imax, ang_na, margin)
		Ibnn = Inmax(Ib_p, tb, Imax, ang_nb, margin)
		Icnn = Inmax(Ic_p, tc, Imax, ang_nc, margin)

		Inn = min(abs(Iann), abs(Ibnn), abs(Icnn)) * np.exp(1j * (np.angle(Vn) + np.pi / 2))
		Ipp = Ip * np.exp(1j * (np.angle(Vp) - np.pi / 2))

		print(Ipp)
		print(Inn)

		Iconv_012 = [0, Ipp, Inn]
		Iconv_abc = x012_to_abc(Iconv_012)

	# end loop

	Ip1_1 = Ipp * np.exp(-1j * np.angle(Vp))
	In1_1 = Inn * np.exp(-1j * np.angle(Vn))
	return [Ip1_1, In1_1, abs(Vp), abs(Vn)]



