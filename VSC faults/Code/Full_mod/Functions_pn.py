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


def Ipmax(I_n, t, Imax, ang_p, margin):
	v_ang_p = np.angle(ang_p)

	Ian_re = np.real(I_n)
	Ian_im = np.imag(I_n)

	aa = 1 + t ** 2
	bb = 2 * Ian_re + 2 * t * Ian_im
	cc = Ian_re ** 2 + Ian_im ** 2 - Imax ** 2

	Iap_11 = (-bb + np.sqrt(bb ** 2 - 4 * aa * cc)) / (2 * aa)
	Iap_22 = (-bb - np.sqrt(bb ** 2 - 4 * aa * cc)) / (2 * aa)

	Iap_re1 = Iap_11
	Iap_re2 = Iap_22
	Iap_im1 = t * Iap_re1
	Iap_im2 = t * Iap_re2
	Ia_p1 = Iap_re1 + Iap_im1 * 1j
	Ia_p2 = Iap_re2 + Iap_im2 * 1j

	if v_ang_p - margin < np.angle(Ia_p1) and v_ang_p + margin > np.angle(Ia_p1):
		I_p = Ia_p1
	else:
		I_p = Ia_p2

	return I_p


def CurrentP_data(Vp, Ip, Vn, phase):

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


def CurrentN_data(Vn, In, Vp, phase):

	ang_Vn = np.angle(Vn)
	In = In * np.exp(1j * (ang_Vn + np.pi / 2)) 

	ang_Vp = np.exp(1j * np.angle(Vp))
	a = np.exp(1j * 2 * np.pi / 3)

	ang_pa = ang_Vp * np.exp(- 1j * np.pi / 2)
	ang_pb = ang_pa * a ** 2
	ang_pc = ang_pa * a

	ta = np.tan(np.angle(ang_pa))
	tb = np.tan(np.angle(ang_pb))
	tc = np.tan(np.angle(ang_pc))

	Ia_n = In
	Ib_n = In * a
	Ic_n = In * a ** 2

	if phase == 'a':
		I_n = Ia_n
		t = ta
		ang_p = ang_pa
	elif phase == 'b':
		I_n = Ib_n
		t = tb
		ang_p = ang_pb
	elif phase == 'c':
		I_n = Ic_n
		t = tc
		ang_p = ang_pc

	return I_n, t, ang_p

