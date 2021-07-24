import numpy as np
import pandas as pd

# Inputs
# bus_data = pd.read_csv('Datafiles/IEEE9.txt')
# bus_slack = 1 - 1
# index_branch_slack = 0
# Zfault = 0.1
# type_fault = '3x'
# bus_ff = 8  # original bus with the fault


def Z_IEEE9(file, Zfault, type_fault, bus_ff):

	# Input data, I know what the system IEEE looks like
	bus_slack = 1 - 1
	index_branch_slack = 0

	# Arguments
	bus_data = pd.read_csv(file)
	Zfault = Zfault
	type_fault = type_fault
	bus_ff = bus_ff


	# Calculations

	# build Y with R/X
	#Y_all = A * Ylist * A^T
	n_br = len(bus_data)
	bu_list = bus_data.loc[:, 'bus i':'bus j'].values
	bu_set = set([item for sublist in bu_list for item in sublist])
	n_bu = len(bu_set)

	# build A
	I3xp = np.eye(3)
	I3xn = - np.eye(3)
	A = np.zeros((3 * n_bu, 3 * n_br))
	for j in range(n_br):
		bus_ii = bus_data['bus i'][j] - 1 
		bus_jj = bus_data['bus j'][j] - 1
		A[3 * bus_ii : 3 * bus_ii + 3, 3 * j : 3 * j + 3] = I3xp 
		A[3 * bus_jj : 3 * bus_jj + 3, 3 * j : 3 * j + 3] = I3xn

	# build Ylist
	Ylist = np.zeros((3 * n_br, 3 * n_br), dtype=complex)
	for j in range(n_br):
		Rr = bus_data['R'][j]
		Xx = bus_data['X'][j]
		Zz = Rr + 1j * Xx
		Yy = 1 / Zz
		Ylist[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] = Yy * I3xp

	Yrx_all = np.dot(A, np.dot(Ylist, np.transpose(A)))
	# Yrx_allpd = pd.DataFrame(Yrx_all)
	# Yrx_allpd.to_excel('Datafiles/Yrx.xlsx')


	# add jB/2 in parallel
	Ysh = np.zeros((3 * n_bu, 3 * n_bu), dtype=complex)
	# fill this and then merge with Yrx
	for j in range(n_br):
		bus_ii = bus_data['bus i'][j] - 1
		bus_jj = bus_data['bus j'][j] - 1
		Bb = bus_data['B'][j]
		Ycap = 1j * Bb / 2
		Ysh[3 * bus_ii : 3 * bus_ii + 3, 3 * bus_ii : 3 * bus_ii + 3] += Ycap * I3xp 
		Ysh[3 * bus_jj : 3 * bus_jj + 3, 3 * bus_jj : 3 * bus_jj + 3] += Ycap * I3xp 

	# Ysh_pd = pd.DataFrame(Ysh)
	# Ysh_pd.to_excel('Datafiles/Ysh.xlsx')

	Yfull_v1 = Yrx_all + Ysh
	# Yfull_pd = pd.DataFrame(Yfull_v1)
	# Yfull_pd.to_excel('Datafiles/Yfull_v1.xlsx')

	# remove its 3 rows and columns
	bus_slack = 1 - 1
	Yfinal_v1 = np.delete(Yfull_v1, [3 * bus_slack, 3 * bus_slack + 1, 3 * bus_slack + 2], 0)
	Yfinal = np.delete(Yfinal_v1, [3 * bus_slack, 3 * bus_slack + 1, 3 * bus_slack + 2], 1)
	# Yfinal_pd = pd.DataFrame(Yfinal)
	# Yfinal_pd.to_excel('Datafiles/Yfinal.xlsx')

	# current for the Norton eq.
	index_branch_slack = 0
	Rt = bus_data['R'][index_branch_slack]
	Xt = bus_data['X'][index_branch_slack]
	Zt = Rt + 1j * Xt
	Vt = 1.04  # or set it to 1.00?
	a = 1 * np.exp(1j * np.pi * 120 / 180)
	Vta = Vt 
	Vtb = Vt * a ** 2
	Vtc = Vt * a
	Ita = Vta / Zt
	Itb = Vtb / Zt
	Itc = Vtc / Zt
	Ig_v = [Ita, Itb, Itc]

	# Cause the fault
	Yfault = 1 / Zfault

	if type_fault == '3x':
		Yff = np.array([[2 * Yfault, - Yfault, - Yfault], [- Yfault, 2 * Yfault, - Yfault], [- Yfault, - Yfault, 2 * Yfault]])  # 3x3 and add
	elif type_fault == 'LG':
		Yff = np.array([[Yfault, 0, 0], [0, 0, 0], [0, 0, 0]])
	elif type_fault == 'LL':
		Yff = np.array([[Yfault, - Yfault, 0], [- Yfault, Yfault, 0], [0, 0, 0]])
	elif type_fault == 'LLG':
		Ymax = 1e10
		Yff = np.array([[Ymax, - Ymax, 0], [-Ymax, Ymax + Yfault, 0], [0, 0, 0]])

	bus_fault = bus_ff - 2
	Yfinal[3 * bus_fault : 3 * bus_fault + 3, 3 * bus_fault : 3 * bus_fault + 3] += Yff

	# Yff_final_pd = pd.DataFrame(Yfinal)
	# Yff_final_pd.to_excel('Datafiles/Yff_final.xlsx')

	# Final impedance matrix
	Zmat = np.linalg.inv(Yfinal)

	return Zmat, 