import numpy as np

# program functions of the lagrangian, solve with NR like in Meseguer, check that the max I is correct


# functions of the absolute value of the currents abc

def fIa(Ipre, Ipim, Inre, Inim):
    valor = Ipre ** 2 + Inre ** 2 + Ipim ** 2 + Inim ** 2 + 2 * Ipre * Inre - 2 * Ipim * Inim
    return np.sqrt(valor)


def fIb(Ipre, Ipim, Inre, Inim):
    valor = Ipre ** 2 + Inre ** 2 + Ipim ** 2 + Inim ** 2 - Ipre * Inre + np.sqrt(3) * Ipre * Inim + np.sqrt(3) * Inre * Ipim + Ipim * Inim
    return np.sqrt(valor)


def fIc(Ipre, Ipim, Inre, Inim):
    valor = Ipre ** 2 + Inre ** 2 + Ipim ** 2 + Inim ** 2 - Ipre * Inre - np.sqrt(3) * Ipre * Inim - np.sqrt(3) * Inre * Ipim + Ipim * Inim
    return np.sqrt(valor)


# functions of the partial derivatives of the Lagrangian for Ia max

def dLIAdIpre(Ipre, Ipim, Inre, Inim, lam):
    valor = lam1 * R + lam1 * 1 / 2 * (Vgp ** 2 - X ** 2 * Ipre ** 2 - R ** 2 * Ipim ** 2 + 2 * X * R * Ipre * Ipim) ** (- 1 / 2) * (- 2 * X ** 2 * Ipre + 2 * X * R * Ipim) - 2 * lam * Ipre - 2 * lam * Inre
    return valor


def dLIAdIpim(Ipre, Ipim, Inre, Inim, lam):
    valor = lam1 * X + lam1 * 1 / 2 * (Vgp ** 2 - X ** 2 * Ipre ** 2 - R ** 2 * Ipim ** 2 + 2 * X * R * Ipre * Ipim) ** (- 1 / 2) * (- 2 * R ** 2 * Ipim + 2 * X * R * Ipre) - 2 * lam * Ipim + 2 * lam * Inim
    return valor


def dLIAdInre(Ipre, Ipim, Inre, Inim, lam):
    valor = - lam2 * R - lam2 * 1 / 2 * (Vgn ** 2 - R ** 2 * Inim ** 2 - X ** 2 * Inre ** 2 - 2 * X * R * Inre * Inim) ** (- 1 / 2) * (- 2 * X ** 2 * Inre - 2 * X * R * Inim) - 2 * lam * Inre - 2 * lam * Ipre
    return valor


def dLIAdInim(Ipre, Ipim, Inre, Inim, lam):
    valor = lam2 * X - lam2 * 1 / 2 * (Vgn ** 2 - R ** 2 * Inim ** 2 - X ** 2 * Inre ** 2 - 2 * X * R * Inre * Inim) ** (- 1 / 2) * (- 2 * R ** 2 * Inim - 2 * X * R * Inre) - 2 * lam * Inim + 2 * lam * Ipim
    return valor


def dLIAdLam(Ipre, Ipim, Inre, Inim, lam):
    valor = - (Ipre ** 2 + Inre ** 2 + Ipim ** 2 + Inim ** 2 + 2 * Ipre * Inre - 2 * Ipim * Inim) + Imax ** 2
    return valor


# functions of the partial derivatives of the Lagrangian for Ib max

def dLIBdIpre(Ipre, Ipim, Inre, Inim, lam):
    valor = lam1 * R + lam1 * 1 / 2 * (Vgp ** 2 - X ** 2 * Ipre ** 2 - R ** 2 * Ipim ** 2 + 2 * X * R * Ipre * Ipim) ** (- 1 / 2) * (- 2 * X ** 2 * Ipre + 2 * X * R * Ipim) - 2 * lam * Ipre + lam * Inre - np.sqrt(3) * lam * Inim
    return valor


def dLIBdIpim(Ipre, Ipim, Inre, Inim, lam):
    valor = lam1 * X + lam1 * 1 / 2 * (Vgp ** 2 - X ** 2 * Ipre ** 2 - R ** 2 * Ipim ** 2 + 2 * X * R * Ipre * Ipim) ** (- 1 / 2) * (- 2 * R ** 2 * Ipim + 2 * X * R * Ipre) - 2 * lam * Ipim - np.sqrt(3) * lam * Inre - lam * Inim
    return valor


def dLIBdInre(Ipre, Ipim, Inre, Inim, lam):
    valor = - lam2 * R - lam2 * 1 / 2 * (Vgn ** 2 - R ** 2 * Inim ** 2 - X ** 2 * Inre ** 2 - 2 * X * R * Inre * Inim) ** (- 1 / 2) * (- 2 * X ** 2 * Inre - 2 * X * R * Inim) - 2 * lam * Inre + lam * Ipre - np.sqrt(3) * lam * Ipim
    return valor


def dLIBdInim(Ipre, Ipim, Inre, Inim, lam):
    valor = lam2 * X - lam2 * 1 / 2 * (Vgn ** 2 - R ** 2 * Inim ** 2 - X ** 2 * Inre ** 2 - 2 * X * R * Inre * Inim) ** (- 1 / 2) * (- 2 * R ** 2 * Inim - 2 * X * R * Inre) - 2 * lam * Inim - np.sqrt(3) * lam * Ipre - lam * Ipim
    return valor


def dLIBdLam(Ipre, Ipim, Inre, Inim, lam):
    valor = - (Ipre ** 2 + Ipim ** 2 + Inre ** 2 + Inim ** 2 - Ipre * Inre + np.sqrt(3) * Ipre * Inim + np.sqrt(3) * Inre * Ipim + Ipim * Inim) + Imax ** 2
    return valor


# functions of the partial derivatives of the Lagrangian for Ic max

def dLICdIpre(Ipre, Ipim, Inre, Inim, lam):
    valor = lam1 * R + lam1 * 1 / 2 * (Vgp ** 2 - X ** 2 * Ipre ** 2 - R ** 2 * Ipim ** 2 + 2 * X * R * Ipre * Ipim) ** (- 1 / 2) * (- 2 * X ** 2 * Ipre + 2 * X * R * Ipim) - 2 * lam * Ipre + lam * Inre + lam * np.sqrt(3) * Inim 
    return valor


def dLICdIpim(Ipre, Ipim, Inre, Inim, lam):
    valor = lam1 * X + lam1 * 1 / 2 * (Vgp ** 2 - X ** 2 * Ipre ** 2 - R ** 2 * Ipim ** 2 + 2 * X * R * Ipre * Ipim) ** (- 1 / 2) * (- 2 * R ** 2 * Ipim + 2 * X * R * Ipre) - 2 * lam * Ipim + np.sqrt(3) * lam * Inre - lam * Inim
    return valor


def dLICdInre(Ipre, Ipim, Inre, Inim, lam):
    valor = - lam2 * R - lam2 * 1 / 2 * (Vgn ** 2 - R ** 2 * Inim ** 2 - X ** 2 * Inre ** 2 - 2 * X * R * Inre * Inim) ** (- 1 / 2) * (- 2 * X ** 2 * Inre - 2 * X * R * Inim) - 2 * lam * Inre + lam * Ipre + np.sqrt(3) * lam * Ipim
    return valor


def dLICdInim(Ipre, Ipim, Inre, Inim, lam):
    valor = lam2 * X - lam2 * 1 / 2 * (Vgn ** 2 - R ** 2 * Inim ** 2 - X ** 2 * Inre ** 2 - 2 * X * R * Inre * Inim) ** (- 1 / 2) * (- 2 * R ** 2 * Inim - 2 * X * R * Inre) - 2 * lam * Inim + np.sqrt(3) * lam * Ipre - lam * Ipim
    return valor


def dLICdLam(Ipre, Ipim, Inre, Inim, lam):
    valor = - (Ipre ** 2 + Ipim ** 2 + Inre ** 2 + Inim ** 2 - Ipre * Inre - np.sqrt(3) * Ipre * Inim - np.sqrt(3) * Inre * Ipim + Ipim * Inim) + Imax ** 2
    return valor


R = 0.01
X = 0.1
Imax = 1

Vgp = 0.8
Vgn = 0.1 

lam1 = 1
lam2 = 1

J = np.zeros((5, 5), dtype = complex)  # lam can take complex values!? Maybe
Aincr = 1e-9  # small value to diferentiate
n_iter = 10

# I initialize the unknowns with this
Ipre = 0.0
Ipim = 0.5
Inre = 0.0
Inim = 0.54
lam = 0.05

for i in range(n_iter):

    Iaf = fIa(Ipre, Ipim, Inre, Inim)
    Ibf = fIb(Ipre, Ipim, Inre, Inim)
    Icf = fIc(Ipre, Ipim, Inre, Inim)

    Imaxx = max(Iaf, Ibf, Icf)

    if Iaf == Imaxx:
        print('Iaaa')
        J[0, 0] = (dLIAdIpre(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLIAdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[0, 1] = (dLIAdIpre(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLIAdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[0, 2] = (dLIAdIpre(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLIAdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[0, 3] = (dLIAdIpre(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLIAdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[0, 4] = (dLIAdIpre(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLIAdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        J[1, 0] = (dLIAdIpim(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLIAdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[1, 1] = (dLIAdIpim(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLIAdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[1, 2] = (dLIAdIpim(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLIAdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[1, 3] = (dLIAdIpim(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLIAdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[1, 4] = (dLIAdIpim(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLIAdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        J[2, 0] = (dLIAdInre(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLIAdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[2, 1] = (dLIAdInre(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLIAdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[2, 2] = (dLIAdInre(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLIAdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[2, 3] = (dLIAdInre(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLIAdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[2, 4] = (dLIAdInre(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLIAdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        J[3, 0] = (dLIAdInim(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLIAdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[3, 1] = (dLIAdInim(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLIAdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[3, 2] = (dLIAdInim(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLIAdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[3, 3] = (dLIAdInim(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLIAdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[3, 4] = (dLIAdInim(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLIAdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        J[4, 0] = (dLIAdLam(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLIAdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[4, 1] = (dLIAdLam(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLIAdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[4, 2] = (dLIAdLam(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLIAdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[4, 3] = (dLIAdLam(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLIAdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[4, 4] = (dLIAdLam(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLIAdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        f1 = dLIAdIpre(Ipre, Ipim, Inre, Inim, lam)
        f2 = dLIAdIpim(Ipre, Ipim, Inre, Inim, lam)
        f3 = dLIAdInre(Ipre, Ipim, Inre, Inim, lam)
        f4 = dLIAdInim(Ipre, Ipim, Inre, Inim, lam)
        f5 = dLIAdLam(Ipre, Ipim, Inre, Inim, lam)
        f = np.array([f1, f2, f3, f4, f5])

        print(J)
        print(f)

        Jinv = np.linalg.inv(J)

        Ax = - np.dot(Jinv, f)

        Ipre += Ax[0]
        Ipim += Ax[1]
        Inre += Ax[2]
        Inim += Ax[3]
        lam += Ax[4]


    elif Ibf == Imaxx:
        print('Ibbb')
        J[0, 0] = (dLIBdIpre(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLIBdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[0, 1] = (dLIBdIpre(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLIBdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[0, 2] = (dLIBdIpre(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLIBdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[0, 3] = (dLIBdIpre(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLIBdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[0, 4] = (dLIBdIpre(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLIBdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        J[1, 0] = (dLIBdIpim(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLIBdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[1, 1] = (dLIBdIpim(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLIBdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[1, 2] = (dLIBdIpim(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLIBdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[1, 3] = (dLIBdIpim(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLIBdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[1, 4] = (dLIBdIpim(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLIBdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        J[2, 0] = (dLIBdInre(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLIBdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[2, 1] = (dLIBdInre(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLIBdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[2, 2] = (dLIBdInre(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLIBdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[2, 3] = (dLIBdInre(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLIBdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[2, 4] = (dLIBdInre(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLIBdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        J[3, 0] = (dLIBdInim(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLIBdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[3, 1] = (dLIBdInim(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLIBdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[3, 2] = (dLIBdInim(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLIBdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[3, 3] = (dLIBdInim(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLIBdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[3, 4] = (dLIBdInim(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLIBdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        J[4, 0] = (dLIBdLam(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLIBdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[4, 1] = (dLIBdLam(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLIBdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[4, 2] = (dLIBdLam(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLIBdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[4, 3] = (dLIBdLam(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLIBdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[4, 4] = (dLIBdLam(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLIBdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        f1 = dLIBdIpre(Ipre, Ipim, Inre, Inim, lam)
        f2 = dLIBdIpim(Ipre, Ipim, Inre, Inim, lam)
        f3 = dLIBdInre(Ipre, Ipim, Inre, Inim, lam)
        f4 = dLIBdInim(Ipre, Ipim, Inre, Inim, lam)
        f5 = dLIBdLam(Ipre, Ipim, Inre, Inim, lam)
        f = np.array([f1, f2, f3, f4, f5])
        Jinv = np.linalg.inv(J)

        Ax = - np.dot(Jinv, f)

        Ipre += Ax[0]
        Ipim += Ax[1]
        Inre += Ax[2]
        Inim += Ax[3]
        lam += Ax[4]


    elif Icf == Imaxx:
        print('Iccc')
        J[0, 0] = (dLICdIpre(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLICdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[0, 1] = (dLICdIpre(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLICdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[0, 2] = (dLICdIpre(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLICdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[0, 3] = (dLICdIpre(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLICdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[0, 4] = (dLICdIpre(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLICdIpre(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        J[1, 0] = (dLICdIpim(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLICdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[1, 1] = (dLICdIpim(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLICdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[1, 2] = (dLICdIpim(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLICdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[1, 3] = (dLICdIpim(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLICdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[1, 4] = (dLICdIpim(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLICdIpim(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        J[2, 0] = (dLICdInre(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLICdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[2, 1] = (dLICdInre(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLICdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[2, 2] = (dLICdInre(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLICdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[2, 3] = (dLICdInre(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLICdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[2, 4] = (dLICdInre(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLICdInre(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        J[3, 0] = (dLICdInim(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLICdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[3, 1] = (dLICdInim(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLICdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[3, 2] = (dLICdInim(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLICdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[3, 3] = (dLICdInim(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLICdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[3, 4] = (dLICdInim(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLICdInim(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        J[4, 0] = (dLICdLam(Ipre + Aincr, Ipim, Inre, Inim, lam) - dLICdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[4, 1] = (dLICdLam(Ipre, Ipim + Aincr, Inre, Inim, lam) - dLICdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[4, 2] = (dLICdLam(Ipre, Ipim, Inre + Aincr, Inim, lam) - dLICdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[4, 3] = (dLICdLam(Ipre, Ipim, Inre, Inim + Aincr, lam) - dLICdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr
        J[4, 4] = (dLICdLam(Ipre, Ipim, Inre, Inim, lam + Aincr) - dLICdLam(Ipre, Ipim, Inre, Inim, lam)) / Aincr

        f1 = dLICdIpre(Ipre, Ipim, Inre, Inim, lam)
        f2 = dLICdIpim(Ipre, Ipim, Inre, Inim, lam)
        f3 = dLICdInre(Ipre, Ipim, Inre, Inim, lam)
        f4 = dLICdInim(Ipre, Ipim, Inre, Inim, lam)
        f5 = dLICdLam(Ipre, Ipim, Inre, Inim, lam)
        f = np.array([f1, f2, f3, f4, f5])
        Jinv = np.linalg.inv(J)

        Ax = - np.dot(Jinv, f)

        Ipre += Ax[0]
        Ipim += Ax[1]
        Inre += Ax[2]
        Inim += Ax[3]
        lam += Ax[4]

    print(Ipre, Ipim, Inre, Inim, lam)



