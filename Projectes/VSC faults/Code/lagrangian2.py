import numpy as np
import pandas as pd

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


def fVcp(Ipre, Ipim, Inre, Inim):
    valor = R * Ipre + X * Ipim + np.sqrt(Vgp ** 2 - (X * Ipre - R * Ipim) ** 2)
    return valor


def fVcn(Ipre, Ipim, Inre, Inim):
    if (Vgn ** 2 - (R * Inim + X * Inre) ** 2) < 0:
        return 1
    else:
        valor = R * Inre - X * Inim + np.sqrt(Vgn ** 2 - (R * Inim + X * Inre) ** 2)
        return valor


R = 0.05
X = 0.1
Imax = 1

Vgp = 0.8
Vgn = 0.1 

vec_Vcp = []
vec_Vcn = []
vec_VcpVcn = []
vec_Imax = []

vec_Ipre = []
vec_Ipim = []
vec_Inre = []
vec_Inim = []

n_iter = 50
n_punts = n_iter ** 4
n_compt = 0

for aa in range(n_iter):
    Ipre = aa / n_iter
    for bb in range(n_iter):
        Ipim = bb / n_iter
        for cc in range(n_iter):
            Inre = cc / n_iter
            for dd in range(n_iter):
                Inim = dd / n_iter

                Iaf = fIa(Ipre, Ipim, Inre, Inim)
                Ibf = fIb(Ipre, Ipim, Inre, Inim)
                Icf = fIc(Ipre, Ipim, Inre, Inim)
                Imaxx = max(Iaf, Ibf, Icf)

                if Imaxx < Imax:
                    vec_Imax.append(Imaxx)

                    Vcp = fVcp(Ipre, Ipim, Inre, Inim)
                    Vcn = fVcn(Ipre, Ipim, Inre, Inim)
                    vec_Vcp.append(Vcp)
                    vec_Vcn.append(Vcn)
                    vec_VcpVcn.append(Vcp - Vcn)

                    vec_Ipre.append(Ipre)
                    vec_Ipim.append(Ipim)
                    vec_Inre.append(Inre)
                    vec_Inim.append(Inim)
                n_compt += 1
                print(n_compt * 100 / n_punts)



maxVcp = max(vec_Vcp)
indVcp = vec_Vcp.index(maxVcp)
print(indVcp)

minVcn = min(vec_Vcn)
indVcn = vec_Vcn.index(minVcn)
print(indVcn)

maxVcpVcn = max(vec_VcpVcn)
indVcpVcn = vec_VcpVcn.index(maxVcpVcn)
print(indVcpVcn)

print(vec_Ipre[indVcpVcn])
print(vec_Ipim[indVcpVcn])
print(vec_Inre[indVcpVcn])
print(vec_Inim[indVcpVcn])

print(vec_Imax[indVcpVcn])

# Ipp = -0.5425 * 1j
# Inn = 0.6115 * 1j

Vcpp = vec_Vcp[indVcpVcn]
Vcnn = vec_Vcn[indVcpVcn]
print(Vcpp, Vcnn)

