# embedding with Q(s), to check Sigma

import numpy as np
import matplotlib.pyplot as plt

# DEFINITIONS 
uth = 1
zth = 0.01 + 0.1 * 1j
pref = 0.5
uref = 1
zvsc = 0.1j
W = 1 
Z0 = 0.4
Yth = 1 / zth
Ythre = np.real(Yth)
Ythim = np.imag(Yth)
Yo = 1 / Z0 

prof = 30
V1re = np.zeros(prof)
V1im = np.zeros(prof)
Q = np.zeros(prof)
V1 = np.zeros(prof, dtype=complex)
X1re = np.zeros(prof)
X1im = np.zeros(prof)
X1 = np.zeros(prof, dtype=complex)


# CALCULATION

# terms [0]
c = 0
V1re[0] = 1
V1im[0] = 0
V1[0] = V1re[0] + V1im[0] * 1j
Q[0] = 0
X1re[0] = 1
X1im[0] = 0
X1[0] = X1re[0] + X1im[0] * 1j


def conv(s1, s2, lim_i, lim_s, r):
    suma = 0
    for k in range(lim_i, lim_s + 1):
        suma += s1[k] * s2[c - k - r]
    return suma


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


def thevenin(U, X):
    """
    U: vector de coeficients de tensió
    X: vector de coeficients de la tensió inversa conjugada
    """

    # complex_type = nb.complex128
    n = len(U)

    r_3 = np. zeros(n, dtype=complex)
    r_2 = np. zeros(n, dtype=complex)
    r_1 = np. zeros(n, dtype=complex)
    r_0 = np. zeros(n, dtype=complex)

    T_03 = np. zeros(n, dtype=complex)
    T_02 = np. zeros(n, dtype=complex)
    T_01 = np. zeros(n, dtype=complex)
    T_00 = np. zeros(n, dtype=complex)
    T_13 = np. zeros(n, dtype=complex)
    T_12 = np. zeros(n, dtype=complex)
    T_11 = np. zeros(n, dtype=complex)
    T_10 = np. zeros(n, dtype=complex)
    T_23 = np. zeros(n, dtype=complex)
    T_22 = np. zeros(n, dtype=complex)
    T_21 = np. zeros(n, dtype=complex)
    T_20 = np. zeros(n, dtype=complex)


    r_0[0] = -1  # inicialització de residus
    r_1[0:n - 1] = U[1:n] / U[0]
    r_2[0:n - 2] = U[2:n] / U[0] - U[1] * np.conj(U[0]) / U[0] * X[1:n - 1]

    T_00[0] = -1  # inicializació de polinomis
    T_01[0] = -1
    T_02[0] = -1
    T_10[0] = 0
    T_11[0] = 1 / U[0]
    T_12[0] = 1 / U[0]
    T_20[0] = 0
    T_21[0] = 0
    T_22[0] = -U[1] * np.conj(U[0]) / U[0]

    for l in range(n):  # càlculs successius
        a = (r_2[0] * r_1[0]) / (- r_0[1] * r_1[0] + r_0[0] * r_1[1] - r_0[0] * r_2[0])
        b = -a * r_0[0] / r_1[0]
        c = 1 - b
        T_03[0] = b * T_01[0] + c * T_02[0]
        T_03[1:n] = a * T_00[0:n - 1] + b * T_01[1:n] + c * T_02[1:n]
        T_13[0] = b * T_11[0] + c * T_12[0]
        T_13[1:n] = a * T_10[0:n - 1] + b * T_11[1:n] + c * T_12[1:n]
        T_23[0] = b * T_21[0] + c * T_22[0]
        T_23[1:n] = a * T_20[0:n - 1] + b * T_21[1:n] + c * T_22[1:n]
        r_3[0:n-2] = a * r_0[2:n] + b * r_1[2:n] + c * r_2[1:n - 1]

        if l == n - 1:  # si és l'última iteració
            t_0 = T_03
            t_1 = T_13
            t_2 = T_23

        r_0[:] = r_1[:]  # actualització de residus
        r_1[:] = r_2[:]
        r_2[:] = r_3[:]

        T_00[:] = T_01[:]  # actualització de polinomis
        T_01[:] = T_02[:]
        T_02[:] = T_03[:]
        T_10[:] = T_11[:]
        T_11[:] = T_12[:]
        T_12[:] = T_13[:]
        T_20[:] = T_21[:]
        T_21[:] = T_22[:]
        T_22[:] = T_23[:]

        r_3 = np.zeros(n, dtype=complex)
        T_03 = np.zeros(n, dtype=complex)
        T_13 = np.zeros(n, dtype=complex)
        T_23 = np.zeros(n, dtype=complex)

    usw = -np.sum(t_0) / np.sum(t_1)
    sth = -np.sum(t_2) / np.sum(t_1)

    sigma_bo = sth / (usw * np.conj(usw))

    u = 0.5 + np.sqrt(0.25 + np.real(sigma_bo) - np. imag(sigma_bo)**2) + np.imag(sigma_bo)*1j  # branca estable
    #u = 0.5 - np.sqrt(0.25 + np.real(sigma_bo) - np.imag(sigma_bo) ** 2) + np.imag(sigma_bo) * 1j  # branca inestable

    ufinal = u * usw  # resultat final

    return ufinal


def Sigma(coeff_matU, coeff_matX, ordre, V_slack):
    """
    coeff_matU: matriu de coeficients de tensió
    coeff_matX: matriu de coeficients de la tensió inversa conjugada
    ordre: profunditat seleccionada
    V_slack: tensions dels busos oscil·lants
    """

    if len(V_slack) > 1:
        print('Els valors poden no ser correctes')

    V0 = V_slack[0]  # tensió del bus oscil·lant de referència
    coeff_A = np.copy(coeff_matU)  # adaptar els coeficients per a la funció racional
    coeff_B= np.copy(coeff_matX)

    coeff_A[0, :] = 1
    for i in range(1, coeff_matU.shape[0]):
        coeff_A[i, :] = coeff_matU[i, :] - (V0 - 1) * coeff_A[i-1, :]
    coeff_B[0, :] = 1
    for i in range(1, coeff_matX.shape[0]):
        coeff_B[i, :] = coeff_matX[i, :] + (V0 - 1) * coeff_matX[i-1, :]

    nbus = coeff_matU.shape[1]
    sigmes = np.zeros(nbus, dtype=complex)

    if ordre % 2 == 0:
        M = int(ordre / 2) - 1
    else:
        M = int(ordre / 2)

    for d in range(nbus):  # emplenar objectes del sistema d'equacions
        a = coeff_A[1:2 * M + 2, d]
        b = coeff_B[0:2 * M + 1, d]
        C = np.zeros((2 * M + 1, 2 * M + 1), dtype=complex)  # matriu del sistema
        for i in range(2 * M + 1):
            if i < M:
                C[1 + i:, i] = a[:2 * M - i]
            else:
                C[i - M:, i] = - b[:3 * M - i + 1]

        lhs = np.linalg.solve(C, -a)
        sigmes[d] = np.sum(lhs[M:])/(np.sum(lhs[:M]) + 1)

    return sigmes



def Sigma2(coeff_matU, coeff_matX, ordre, V_slack):
    """
    coeff_matU: matriu de coeficients de tensió
    coeff_matX: matriu de coeficients de la tensió inversa conjugada
    ordre: profunditat seleccionada
    V_slack: tensions dels busos oscil·lants
    """

    if len(V_slack) > 1:
        print('Els valors poden no ser correctes')

    V0 = V_slack[0]  # tensió del bus oscil·lant de referència
    coeff_A = np.copy(coeff_matU)  # adaptar els coeficients per a la funció racional
    coeff_B= np.copy(coeff_matX)

    coeff_A[0,] = 1
    for i in range(1, coeff_matU.shape[0]):
        coeff_A[i] = coeff_matU[i] - (V0 - 1) * coeff_A[i-1]
    coeff_B[0] = 1
    for i in range(1, coeff_matX.shape[0]):
        coeff_B[i] = coeff_matX[i] + (V0 - 1) * coeff_matX[i-1]

    nbus = 1
    sigmes = np.zeros(1, dtype=complex)

    if ordre % 2 == 0:
        M = int(ordre / 2) - 1
    else:
        M = int(ordre / 2)

    a = coeff_A[1:2 * M + 2]
    b = coeff_B[0:2 * M + 1]
    C = np.zeros((2 * M + 1, 2 * M + 1), dtype=complex)  # matriu del sistema
    for i in range(2 * M + 1):
        if i < M:
            C[1 + i:, i] = a[:2 * M - i]
        else:
            C[i - M:, i] = - b[:3 * M - i + 1]

    lhs = np.linalg.solve(C, -a)
    sigmes = np.sum(lhs[M:])/(np.sum(lhs[:M]) + 1)

    return sigmes


# terms [1]
c = 1
mat = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
rhs = np.zeros(3)

mat[0,0] = Ythre
mat[0,1] = -Ythim
mat[0,2] = 0

mat[1,0] = Ythim
mat[1,1] = Ythre
mat[1,2] = -1

mat[2,0] = 2
mat[2,1] = 0 
mat[2,2] = 0

mat_inv = np.linalg.inv(mat)

rhs[0] = pref - np.real(Yo * V1[0])
rhs[1] = -np.imag(Yo * V1[0])
rhs[2] = W - 1

lhs = np.dot(mat_inv, rhs)

V1re[1] = lhs[0]
V1im[1] = lhs[1]
V1[1] = V1re[1] + V1im[1] * 1j
Q[1] = lhs[2]
X1[1] = - conv(X1, np.conj(V1), 0, 1-1, 0)
X1re[1] = np.real(X1[1])
X1im[1] = np.imag(X1[1])


# terms [c>=2]
for c in range(2, prof):
    rhs[0] = pref * X1re[c-1] - np.real(Yo * V1[c-1]) + conv(Q, X1im, 1, c-1, 0)
    rhs[1] = pref * X1im[c-1] - np.imag(Yo * V1[c-1]) + conv(Q, X1re, 1, c-1, 0)
    rhs[2] = -conv(V1re, V1re, 1, c-1, 0) - conv(V1im, V1im, 1, c-1, 0)

    lhs = np.dot(mat_inv, rhs)

    V1re[c] = lhs[0]
    V1im[c] = lhs[1]
    Q[c] = lhs[2]
    V1[c] = V1re[c] + V1im[c] * 1j 
    X1[c] = -conv(X1, np.conj(V1), 0, c-1, 0)
    X1re[c] = np.real(X1[c])
    X1im[c] = np.imag(X1[c])

    if c == 2:
        print(rhs)
        print(lhs)
        print(mat)
        print(mat_inv)
        V1f = sum(V1re) + sum(V1im) * 1j
        V1fre = np.real(V1f)
        V1fim = np.imag(V1f)
        Qf = sum(Q)
        Io = V1f / Z0
        It = (1 - V1f) / zth
        Iv = Io - It
        Vvsc = V1f + Iv * zvsc

        print('Io: ', Io)
        print('Iv: ', Iv)
        print('V0: ', V1f)
        print('Vvsc: ', Vvsc)
        print('Q: ', Qf)

        error = (pref - Qf * 1j) / np.conj(V1f) + Yth - V1f * (Yth + Yo)
        print('error: ', error)
        Xf = sum(X1)
        print(Ythre * V1fre - Ythim * V1fim - Ythre + np.real(V1f * Yo) - pref * np.real(Xf) + Qf * np.imag(Xf))
        print(Ythim * V1fre + Ythre * V1fim - Ythim + np.imag(V1f * Yo) - Qf * np.real(Xf) + pref * np.imag(Xf))

V1f = sum(V1re) + sum(V1im) * 1j
V1fre = np.real(V1f)
V1fim = np.imag(V1f)
Qf = sum(Q)
Io = V1f / Z0
It = (1 - V1f) / zth
Iv = Io - It
Vvsc = V1f + Iv * zvsc

print('Io: ', Io)
print('Iv: ', Iv)
print('V0: ', V1f)
print('Vvsc: ', Vvsc)
print('Q: ', Qf)

error = (pref - Qf * 1j) / np.conj(V1f) + Yth - V1f * (Yth + Yo)
print('error: ', error)

Xf = sum(X1)
print(Ythre * V1fre - Ythim * V1fim - Ythre + np.real(V1f * Yo) - pref * np.real(Xf) + Qf * np.imag(Xf))
print(Ythim * V1fre + Ythre * V1fim - Ythim + np.imag(V1f * Yo) - Qf * np.real(Xf) + pref * np.imag(Xf))

print(V1[:5])
print(X1[:5])
print(Q[:5])

# EXTRA:
Vth = thevenin(V1, X1)
print(abs(Vth))

Sig = Sigma2(V1, X1, prof-1, [1])
Sig_re = np.real(Sig)
Sig_im = np.imag(Sig)
print(Sig_re + 1j * Sig_im)

# .......................GRÀFIC SIGMA ........................
a = []
b = []
c = []

x = np.linspace(-0.25, 1, 1000)
y = np.sqrt(0.25 + x)
a.append(x)
b.append(y)
c.append(-y)

plt.plot(np.real(Sig_re), np.real(Sig_im), 'ro', markersize=2)
plt.plot(x, y)
plt.plot(x, -y)
plt.ylabel('Sigma im')
plt.xlabel('Sigma re')
plt.title('Gràfic Sigma')
plt.show()