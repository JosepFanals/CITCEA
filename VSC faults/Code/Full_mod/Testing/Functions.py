import numpy as np

def Inmax(I_p, t, Imax, ang_n, margin):
	v_ang_n = np.angle(ang_n)

	Iap_re = np.real(I_p)
	Iap_im = np.imag(I_p)

	aa = 1 + t ** 2
	bb = 2 * Iap_re + 2 * t * Iap_im
	cc = Iap_re ** 2 + Iap_im ** 2 - Imax ** 2

	Ian_11 = (-bb + np.sqrt(bb ** 2 - 4 * aa * cc)) / (2 * aa)
	Ian_22 = (-bb - np.sqrt(bb ** 2 - 4 * aa * cc)) / (2 * aa)

	Ian_re1 = Ian_11
	Ian_re2 = Ian_22
	Ian_im1 = t * Ian_re1
	Ian_im2 = t * Ian_re2
	Ia_n1 = Ian_re1 + Ian_im1 * 1j
	Ia_n2 = Ian_re2 + Ian_im2 * 1j

	if v_ang_n - margin < np.angle(Ia_n1) and v_ang_n + margin > np.angle(Ia_n1):
		I_n = Ia_n1
	else:
		I_n = Ia_n2

	return I_n


def Current_data(Vp, Ip, Vn, phase):

	ang_Vp = np.angle(Vp)
	Ip = Ip * np.exp(1j * (ang_Vp - np.pi / 2)) 

	ang_Vn = np.exp(1j * np.angle(Vn))
	a = np.exp(1j * 2 * np.pi / 3)

	ang_na = ang_Vn * np.exp(1j * np.pi / 2)
	ang_nb = ang_na * a
	ang_nc = ang_na * a ** 2

	ta = np.tan(np.angle(ang_na))
	tb = np.tan(np.angle(ang_nb))
	tc = np.tan(np.angle(ang_nc))

	Ia_p = Ip
	Ib_p = Ip * a ** 2
	Ic_p = Ip * a 

	if phase == 'a':
		I_p = Ia_p
		t = ta
		ang_n = ang_na
	elif phase == 'b':
		I_p = Ib_p
		t = tb
		ang_n = ang_nb
	elif phase == 'c':
		I_p = Ic_p
		t = tc
		ang_n = ang_nc

	return I_p, t, ang_n
