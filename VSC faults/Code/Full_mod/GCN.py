import numpy as np
from Functions import Ipmax, CurrentN_data

# Voltages to test
Vp = 0.8 * np.exp(1j * 30 / 180 * np.pi)
Vn = 0.3 * np.exp(1j * 65 / 180 * np.pi)


# Input parameters
Imax = 1
Vnabs = np.abs(Vn)
Vnhigh = 0.6
Vnlow = 0.1
kn = 2
margin = 0.01 


# Grid code negative sequence
In = 0
if Vnabs < Vnhigh and Vnabs > Vnlow:
	In = kn * (Vnabs - Vnlow)

elif Vnabs >= Vnhigh:
	In = Imax

Ia_n, ta, ang_pa = CurrentN_data(Vn, In, Vp, 'a')
Ib_n, tb, ang_pb = CurrentN_data(Vn, In, Vp, 'b')
Ic_n, tc, ang_pc = CurrentN_data(Vn, In, Vp, 'c')


# Calculation of max positive sequence current:
Iapp = Ipmax(Ia_n, ta, Imax, ang_pa, margin)
Ibpp = Ipmax(Ib_n, tb, Imax, ang_pb, margin)
Icpp = Ipmax(Ic_n, tc, Imax, ang_pc, margin)

Ipp = min(abs(Iapp), abs(Ibpp), abs(Icpp)) * np.exp(1j * (np.angle(Vp) - np.pi / 2))
Inn = In * np.exp(1j * (np.angle(Vn) + np.pi / 2))

print(Ipp)
print(Inn)


