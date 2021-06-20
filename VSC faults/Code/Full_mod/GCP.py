import numpy as np
from Functions import Inmax, CurrentP_data

# Voltages to test
Vp = 0.8 * np.exp(1j * 30 / 180 * np.pi)
Vn = 0.3 * np.exp(1j * 65 / 180 * np.pi)


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


