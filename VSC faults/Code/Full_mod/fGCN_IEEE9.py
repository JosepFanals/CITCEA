import numpy as np
from Functions_pn import Ipmax, CurrentN_data
from Functions_main import xabc_to_012, x012_to_abc, build_static_objects, build_static_objects1 
from Yadm_IEEE9 import Z_IEEE9
np.set_printoptions(precision=4)


def fGCN_2vsc(V_mod, Imax, Zfault, type_fault, bus_fault, lam_vec):

	# Functions
	def volt_solution(x):
		m1_inv, Ig_v, n_bus = Z_IEEE9('Datafiles/IEEE9.txt', Zfault, type_fault, bus_fault, V_mod)
		# m1_inv = static_objects[0]
		# Ig_v = static_objects[1] 
		# Ii_v = np.zeros((9,1), dtype=complex)
		Ii_v = np.zeros((n_bus * 3, 1), dtype=complex)
		Ii_v[0:3] = [[x[0]], [x[1]], [x[2]]]
		Ii_v[3:6] = [[x[3]], [x[4]], [x[5]]]
		Ii_v[6:9] = [[Ig_v[0]], [Ig_v[1]], [Ig_v[2]]]

		Vv_v = np.dot(m1_inv, Ii_v)
		return Vv_v

	# static_objects = build_static_objects(V_mod, Zv1, Zv2, Zt, Y_con, Y_gnd)

	Iconv_abc_1 = [0, 0, 0]
	Iconv_abc_prev_1 = [1, 1, 1]
	Iconv_abc_2 = [0, 0, 0]
	Iconv_abc_prev_2 = [1, 1, 1]

	tol = 1e-4
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
		Vnabs1 = np.abs(Vn1)
		Vnabs2 = np.abs(Vn2)
		Vnhigh = 0.6
		Vnlow = 0.1
		kn = 2
		margin = 0.0001 


		# Grid code negative sequence
		In1 = 0
		if Vnabs1 < Vnhigh and Vnabs1 > Vnlow:
			In1 = kn * (Vnabs1 - Vnlow)
		elif Vnabs1 >= Vnhigh:
			In1 = Imax

		In2 = 0
		if Vnabs2 < Vnhigh and Vnabs2 > Vnlow:
			In2 = kn * (Vnabs2 - Vnlow)
		elif Vnabs2 >= Vnhigh:
			In2 = Imax

		Ia_n1, ta1, ang_pa1 = CurrentN_data(Vn1, In1, Vp1, 'a')
		Ib_n1, tb1, ang_pb1 = CurrentN_data(Vn1, In1, Vp1, 'b')
		Ic_n1, tc1, ang_pc1 = CurrentN_data(Vn1, In1, Vp1, 'c')

		Ia_n2, ta2, ang_pa2 = CurrentN_data(Vn2, In2, Vp2, 'a')
		Ib_n2, tb2, ang_pb2 = CurrentN_data(Vn2, In2, Vp2, 'b')
		Ic_n2, tc2, ang_pc2 = CurrentN_data(Vn2, In2, Vp2, 'c')

		# Calculation of max positive sequence current:
		Iapp1 = Ipmax(Ia_n1, ta1, Imax, ang_pa1, margin)
		Ibpp1 = Ipmax(Ib_n1, tb1, Imax, ang_pb1, margin)
		Icpp1 = Ipmax(Ic_n1, tc1, Imax, ang_pc1, margin)

		Iapp2 = Ipmax(Ia_n2, ta2, Imax, ang_pa2, margin)
		Ibpp2 = Ipmax(Ib_n2, tb2, Imax, ang_pb2, margin)
		Icpp2 = Ipmax(Ic_n2, tc2, Imax, ang_pc2, margin)

		Ipp1 = min(abs(Iapp1), abs(Ibpp1), abs(Icpp1)) * np.exp(1j * (np.angle(Vp1) - np.pi / 2))
		Inn1 = In1 * np.exp(1j * (np.angle(Vn1) + np.pi / 2))

		Ipp2 = min(abs(Iapp2), abs(Ibpp2), abs(Icpp2)) * np.exp(1j * (np.angle(Vp2) - np.pi / 2))
		Inn2 = In2 * np.exp(1j * (np.angle(Vn2) + np.pi / 2))


		Iconv_012_1 = [0, Ipp1, Inn1]
		Iconv_abc_1 = x012_to_abc(Iconv_012_1)

		Iconv_012_2 = [0, Ipp2, Inn2]
		Iconv_abc_2 = x012_to_abc(Iconv_012_2)


	# end loop

	Ip1_1 = Ipp1 * np.exp(-1j * np.angle(Vp1))
	In1_1 = Inn1 * np.exp(-1j * np.angle(Vn1))

	Ip1_2 = Ipp2 * np.exp(-1j * np.angle(Vp2))
	In1_2 = Inn2 * np.exp(-1j * np.angle(Vn2))


	return [Ip1_1, In1_1, Ip1_2, In1_2, abs(Vp1), abs(Vn1), abs(Vp2), abs(Vn2)]





