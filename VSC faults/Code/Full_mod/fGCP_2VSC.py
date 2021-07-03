import numpy as np
from Functions_pn import Inmax, CurrentP_data
from Functions_main import xabc_to_012, x012_to_abc, build_static_objects, build_static_objects1 
np.set_printoptions(precision=4)


def fGCP_2vsc(V_mod, Imax, Zv1, Zv2, Zt, Y_con, Y_gnd, lam_vec):

	# Functions
	def volt_solution(x):
        	m1_inv = static_objects[0]
        	Ig_v = static_objects[1] 

        	Ii_v = np.zeros((9,1), dtype=complex)
        	Ii_v[0:3] = [[x[0]], [x[1]], [x[2]]]
        	Ii_v[3:6] = [[x[3]], [x[4]], [x[5]]]
        	Ii_v[6:9] = [Ig_v[0], Ig_v[1], Ig_v[2]]

        	Vv_v = np.dot(m1_inv, Ii_v)
        	return Vv_v

	static_objects = build_static_objects(V_mod, Zv1, Zv2, Zt, Y_con, Y_gnd)

	Iconv_abc_1 = [0, 0, 0]
	Iconv_abc_prev_1 = [1, 1, 1]
	Iconv_abc_2 = [0, 0, 0]
	Iconv_abc_prev_2 = [1, 1, 1]

	tol = 1e-1
	compt = 0
	compt_lim = 1000

	# loop
	while (abs(Iconv_abc_1[0] - Iconv_abc_prev_1[0]) > tol or abs(Iconv_abc_1[1] - Iconv_abc_prev_1[1]) > tol or abs(Iconv_abc_1[2] - Iconv_abc_prev_1[2]) > tol or abs(Iconv_abc_2[0] - Iconv_abc_prev_2[0]) > tol or abs(Iconv_abc_2[1] - Iconv_abc_prev_2[1]) > tol or abs(Iconv_abc_2[2] - Iconv_abc_prev_2[2]) > tol) and compt < compt_lim:
		compt += 1
		Iconv_abc_prev_1 = Iconv_abc_1
		Iconv_abc_prev_2 = Iconv_abc_2
		Iconv_abc = [Iconv_abc_1[0], Iconv_abc_1[1], Iconv_abc_1[2], Iconv_abc_2[0], Iconv_abc_2[1], Iconv_abc_2[2]]

		Vv_v = volt_solution(Iconv_abc)
		V_p1_abc = Vv_v[0:3]
		V_p2_abc = Vv_v[3:6]
		V_p1_012 = xabc_to_012(V_p1_abc)
		V_p2_012 = xabc_to_012(V_p2_abc)

		Vp1 = V_p1_012[1]
		Vn1 = V_p1_012[2]

		Vp2 = V_p2_012[1]
		Vn2 = V_p2_012[2]

		# Input parameters
		Imax = 1
		Vpabs1 = np.abs(Vp1)
		Vpabs2 = np.abs(Vp2)
		Vphigh = 0.9
		Vplow = 0.4
		kp = 2
		margin = 0.0001 


		# Grid code positive sequence
		Ip1 = 0
		if Vpabs1 < Vphigh and Vpabs1 > Vplow:
			Ip1 = kp * (Vphigh - Vpabs1)
		elif Vpabs1 <= Vplow:
			Ip1 = Imax

		Ip2 = 0
		if Vpabs2 < Vphigh and Vpabs2 > Vplow:
			Ip2 = kp * (Vphigh - Vpabs2)
		elif Vpabs2 <= Vplow:
			Ip2 = Imax

		Ia_p1, ta1, ang_na1 = CurrentP_data(Vp1, Ip1, Vn1, 'a')
		Ib_p1, tb1, ang_nb1 = CurrentP_data(Vp1, Ip1, Vn1, 'b')
		Ic_p1, tc1, ang_nc1 = CurrentP_data(Vp1, Ip1, Vn1, 'c')

		Ia_p2, ta2, ang_na2 = CurrentP_data(Vp2, Ip2, Vn2, 'a')
		Ib_p2, tb2, ang_nb2 = CurrentP_data(Vp2, Ip2, Vn2, 'b')
		Ic_p2, tc2, ang_nc2 = CurrentP_data(Vp2, Ip2, Vn2, 'c')

		# Calculation of max negative sequence current:
		Iann1 = Inmax(Ia_p1, ta1, Imax, ang_na1, margin)
		Ibnn1 = Inmax(Ib_p1, tb1, Imax, ang_nb1, margin)
		Icnn1 = Inmax(Ic_p1, tc1, Imax, ang_nc1, margin)

		Iann2 = Inmax(Ia_p2, ta2, Imax, ang_na2, margin)
		Ibnn2 = Inmax(Ib_p2, tb2, Imax, ang_nb2, margin)
		Icnn2 = Inmax(Ic_p2, tc2, Imax, ang_nc2, margin)

		Inn1 = min(abs(Iann1), abs(Ibnn1), abs(Icnn1)) * np.exp(1j * (np.angle(Vn1) + np.pi / 2))
		Ipp1 = Ip1 * np.exp(1j * (np.angle(Vp1) - np.pi / 2))

		Inn2 = min(abs(Iann2), abs(Ibnn2), abs(Icnn2)) * np.exp(1j * (np.angle(Vn2) + np.pi / 2))
		Ipp2 = Ip2 * np.exp(1j * (np.angle(Vp2) - np.pi / 2))


		Iconv_012_1 = [0, Ipp1, Inn1]
		Iconv_abc_1 = x012_to_abc(Iconv_012_1)

		Iconv_012_2 = [0, Ipp2, Inn2]
		Iconv_abc_2 = x012_to_abc(Iconv_012_2)

		print(compt)

	# end loop

	Ip1_1 = Ipp1 * np.exp(-1j * np.angle(Vp1))
	In1_1 = Inn1 * np.exp(-1j * np.angle(Vn1))

	Ip1_2 = Ipp2 * np.exp(-1j * np.angle(Vp2))
	In1_2 = Inn2 * np.exp(-1j * np.angle(Vn2))
	

	return [Ip1_1, In1_1, Ip1_2, In1_2, abs(Vp1), abs(Vn1), abs(Vp2), abs(Vn2)]


