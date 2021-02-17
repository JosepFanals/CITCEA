# System 1 from song2021solution. 
# Values match with the rudimentary basic 


import numpy as np
import matplotlib.pyplot as plt


def pade4all(ordre, coeff_mat, s):
    """
    ordre: profunditat seleccionada
    coeff_mat: matriu o vector de coeficients
    s: valor en el qual s'avalua la sèrie , sovint s=1
    """

    if coeff_mat.ndim > 1:  # nombre de columnes
        nbus = coeff_mat.shape[1]
    else:
        nbus = coeff_mat.ndim

    voltatges = np.zeros(nbus, dtype=complex)  # resultats finals

    if ordre % 2 != 0:
        nn = int(ordre / 2)
        L = nn
        M = nn
        for d in range(nbus):
            if nbus > 1:
                rhs = coeff_mat[L + 1:L + M + 1, d]  # vector de la dreta , conegut
            else:
                rhs = coeff_mat[L + 1:L + M + 1]

            C = np.zeros((M, M), dtype=complex)  # matriu del sistema
            for i in range(M):
                k = i + 1
                if nbus > 1:
                    C[i, :] = coeff_mat[L - M + k:L + k, d]
                else:
                    C[i, :] = coeff_mat[L - M + k:L + k]

            b = np.zeros(rhs.shape[0] + 1, dtype=complex)  # denominador
            x = np.linalg.solve(C, -rhs)
            b[0] = 1
            b[1:] = x[::-1]

            a = np.zeros(L + 1, dtype=complex)  # numerador
            if nbus > 1:
                a[0] = coeff_mat[0, d]
            else:
                a[0] = coeff_mat[0]

            for i in range(L):  # completar numerador
                val = complex(0)
                k = i + 1
                for j in range(k + 1):
                    if nbus > 1:
                        val += coeff_mat[k - j, d] * b[j]
                    else:
                        val += coeff_mat[k - j] * b[j]
                a[i + 1] = val

            p = complex(0)
            q = complex(0)

            for i in range(len(a)):  # avaluar numerador i denominador
                p += a[i] * s ** i
            for i in range(len(b)):
                q += b[i] * s ** i

            voltatges[d] = p / q

            ppb = np.poly1d(b)  # convertir a polinomi
            ppa = np.poly1d(a)
            ppbr = ppb.r  # pols
            ppar = ppa.r  # zeros
    else:
        nn = int(ordre / 2)
        L = nn
        M = nn - 1
        for d in range(nbus):
            if nbus > 1:
                rhs = coeff_mat[M + 2: 2 * M + 2, d]  # vector de la dreta , conegut
            else:
                rhs = coeff_mat[M + 2: 2 * M + 2]

            C = np.zeros((M, M), dtype=complex)  # matriu del sistema
            for i in range(M):
                k = i + 1
                if nbus > 1:
                    C[i, :] = coeff_mat[L - M + k:L + k, d]
                else:
                    C[i, :] = coeff_mat[L - M + k:L + k]

            b = np.zeros(rhs.shape[0] + 1, dtype=complex)  # denominador
            x = np.linalg.solve(C, -rhs)
            b[0] = 1
            b[1:] = x[::-1]

            a = np.zeros(L + 1, dtype=complex)  # numerador
            if nbus > 1:
                a[0] = coeff_mat[0, d]
            else:
                a[0] = coeff_mat[0]

            for i in range(1, L):  # completar numerador
                val = complex(0)
                for j in range(i + 1):
                    if nbus > 1:
                        val += coeff_mat[i - j, d] * b[j]
                    else:
                        val += coeff_mat[i - j] * b[j]
                a[i] = val

            val = complex(0)
            for j in range(L):
                if nbus > 1:
                    val += coeff_mat[M - j + 1, d] * b[j]
                else:
                    val += coeff_mat[M - j + 1] * b[j]
            a[L] = val

            p = complex(0)
            q = complex(0)

            for i in range(len(a)):  # avaluar numerador i denominador
                p += a[i] * s ** i
            for i in range(len(b)):
                q += b[i] * s ** i

            voltatges[d] = p / q

            ppb = np.poly1d(b)  # convertir a polinomi
            ppa = np.poly1d(a)
            ppbr = ppb.r  # pols
            ppar = ppa.r  # zeros

    return voltatges


# DEFINITIONS 
uth = 1
zth = 0.01 + 0.1 * 1j
pref = 0.5
uref = 1
zvsc = 0.1j
W = 1 
Z0 = np.arange(0.15, 1.001, 0.001) 
n_test = int((1.001 - 0.15) / 0.001)
Yth = 1 / zth
Ythre = np.real(Yth)
Ythim = np.imag(Yth)
Yo = np.zeros(n_test + 1)
Yo[:] = 1 / Z0[:]

prof = 30
V1re = np.zeros(prof)
V1im = np.zeros(prof)
Vvf = np.zeros(n_test)

# CALCULATION 

for kk in range(n_test):

    # terms [0]
    V1re[0] = 1
    V1im[0] = 0

    # terms [1]
    V1re[1] = (W - 1) / 2
    V1im[1] = (-pref + V1re[1] * Ythre + Yo[kk]) / Ythim

    # terms[2]
    V1re[2] = (-V1re[1] * V1re[1] - V1im[1] * V1im[1]) / 2
    V1im[2] = (Ythre * V1re[2] + Ythre * V1re[1] * V1re[1] + 2 * Yo[kk] * V1re[1] + Ythre * V1im[1] * V1im[1]) / Ythim


    def conv(s1, s2, lim_i, lim_s, r):
        suma = 0
        for k in range(lim_i, lim_s + 1):
            suma += s1[k] * s2[c - k - r]
        return suma


    for c in range(3, prof):
        V1re[c] = (-conv(V1re, V1re, 1, c-1, 0) - conv(V1im, V1im, 1, c-1, 0)) / 2
        V1im[c] = (Ythre * V1re[c] + Ythre * conv(V1re, V1re, 1, c-1, 0) + Yo[kk] * conv(V1re, V1re, 0, c-1, 1) + Ythre * conv(V1im, V1im, 1, c-1, 0) + Yo[kk] * conv(V1im, V1im, 0, c-1, 1)) / Ythim


    # V1f = sum(V1re) + sum(V1im) * 1j
    # Io = V1f / Z0[kk]
    # It = (1 - V1f) / zth
    # Iv = Io - It
    # Vvsc = V1f + Iv * zvsc

    V1f = pade4all(prof - 1, V1re, 1) + pade4all(prof - 1, V1im, 1) * 1j
    Io = V1f / Z0[kk]
    It = (1 - V1f) / zth
    Iv = Io - It
    Vvsc = V1f + Iv * zvsc

    Vvf[kk] = abs(Vvsc)

print('Vvf: ', Vvf)
plt.plot(Z0[:n_test], Vvf)
plt.ylabel('some numbers')
plt.show()