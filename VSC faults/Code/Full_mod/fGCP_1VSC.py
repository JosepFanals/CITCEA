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

	Iconv_abc = [0.2, 0.2, 0.2]
	Iconv_abc_prev = [0.1, 0.1, 0.1]
	Ipp = 0.3 + 0 * 1j
	Inn = 0.5 + 0 * 1j
	Ipp_prev = 0.1 + 0.1 * 1j
	Inn_prev = 0.2 + 0.2 * 1j
	tol = 1e-5
	fr = 0.3  # relaxation factor
	compt = 0
	compt_lim = 100
	trobat = False

	print('begin')
	# loop
	# while (abs(Iconv_abc[0] - Iconv_abc_prev[0]) > tol or abs(Iconv_abc[1] - Iconv_abc_prev[1]) > tol or abs(Iconv_abc[2] - Iconv_abc_prev[2]) > tol) and compt < compt_lim:
	while (abs(Ipp - Ipp_prev) > tol or abs(Inn - Inn_prev) > tol) and compt < compt_lim:
		compt += 1
		# print(Iconv_abc, Iconv_abc_prev)
		# print(Iconv_abc_prev)
		# print(compt)
		Iconv_abc_prev = Iconv_abc
		Vv_v = volt_solution(Iconv_abc)
		V_p1_abc = Vv_v[0:3]
		V_p1_012 = xabc_to_012(V_p1_abc)

		Ipp_prev = Ipp
		Inn_prev = Inn

		Vp = V_p1_012[1]
		Vn = V_p1_012[2]

		# Input parameters
		Imax = 1
		Vpabs = np.abs(Vp)
		Vphigh = 0.9
		Vplow = 0.4
		kp = 2
		margin = 0.001 


		# Grid code positive sequence
		Ip = 0
		if Vpabs < Vphigh and Vpabs > Vplow:
			Ip = kp * (Vphigh - Vpabs)
		elif Vpabs <= Vplow:
			Ip = Imax


		if Ip < Imax:
			Ia_p, ta, ang_na = CurrentP_data(Vp, Ip, Vn, 'a')
			Ib_p, tb, ang_nb = CurrentP_data(Vp, Ip, Vn, 'b')
			Ic_p, tc, ang_nc = CurrentP_data(Vp, Ip, Vn, 'c')


			# Calculation of max negative sequence current:
			Iann = Inmax(Ia_p, ta, Imax, ang_na, margin)
			Ibnn = Inmax(Ib_p, tb, Imax, ang_nb, margin)
			Icnn = Inmax(Ic_p, tc, Imax, ang_nc, margin)

			Inn = min(abs(Iann), abs(Ibnn), abs(Icnn)) * np.exp(1j * (np.angle(Vn) + np.pi / 2))
		
		else:
			Inn = 0

		Ipp = Ip * np.exp(1j * (np.angle(Vp) - np.pi / 2))

		# with relaxation
		Ipp = Ipp_prev + fr * (Ipp - Ipp_prev)
		Inn = Inn_prev + fr * (Inn - Inn_prev)


		Iconv_012 = [0, Ipp, Inn]
		Iconv_abc = x012_to_abc(Iconv_012)
		# print(max(abs(Iconv_abc[0]), abs(Iconv_abc[1]), abs(Iconv_abc[2])))
		# print(abs(Iconv_abc[0]), abs(Iconv_abc[1]), abs(Iconv_abc[2]))
		# print(np.real(Iconv_abc[0]), np.imag(Iconv_abc[0]), np.real(Iconv_abc[1]), np.imag(Iconv_abc[1]), np.real(Iconv_abc[2]), np.imag(Iconv_abc[2]))

	# end loop
	print(compt)

	Ip1_1 = Ipp * np.exp(-1j * np.angle(Vp))
	In1_1 = Inn * np.exp(-1j * np.angle(Vn))
	print(Ip1_1)
	print(In1_1)
	return [Ip1_1, In1_1, abs(Vp), abs(Vn)]



