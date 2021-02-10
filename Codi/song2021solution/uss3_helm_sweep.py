# unsaturated converter

import numpy as np
import matplotlib.pyplot as plt

sig_vec = []
for mm in range(100, 1000, 1):
    Z0 = mm / 1000

    # DEFINITIONS 
    uth = 1
    zth = 0.01 + 0.1 * 1j
    pref = 0.5
    uref = 1
    zvsc = 0.1j
    W = 1 
    # Z0 = 0.15
    Yth = 1 / zth
    Ythre = np.real(Yth)
    Ythim = np.imag(Yth)
    Yo = 1 / Z0 

    prof = 7
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


    # terms [1]
    c = 1
    mat = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
    rhs = np.zeros(3)

    mat[0,0] = Ythre
    mat[0,1] = -Ythim
    mat[0,2] = 0

    mat[1,0] = Ythim
    mat[1,1] = Ythre
    mat[1,2] = 1

    mat[2,0] = 2
    mat[2,1] = 0 
    mat[2,2] = 0

    mat_inv = np.linalg.inv(mat)

    rhs[0] = np.real(pref * X1[0] - Yo * V1[0])
    rhs[1] = np.imag(pref * X1[0] - Yo * V1[0])
    rhs[2] = W - 1

    lhs = np.dot(mat_inv, rhs)

    V1re[1] = lhs[0]
    V1im[1] = lhs[1]
    V1[1] = V1re[1] + V1im[1] * 1j
    Q[1] = lhs[2]
    X1[1] = - np.conj(V1[1])
    X1re[1] = np.real(X1[1])
    X1im[1] = np.imag(X1[1])


    # terms [c>=2]
    for c in range(2, prof):
        rhs[0] = pref * X1re[c-1] - np.real(Yo * V1[c-1]) + conv(Q, X1im, 1, c-1, 0)
        rhs[1] = pref * X1im[c-1] - np.imag(Yo * V1[c-1]) - conv(Q, X1re, 1, c-1, 0)
        rhs[2] = -conv(V1re, V1re, 1, c-1, 0) - conv(V1im, V1im, 1, c-1, 0)

        lhs = np.dot(mat_inv, rhs)

        V1re[c] = lhs[0]
        V1im[c] = lhs[1]
        Q[c] = lhs[2]
        V1[c] = V1re[c] + V1im[c] * 1j 
        X1[c] = -conv(X1, np.conj(V1), 0, c-1, 0)
        X1re[c] = np.real(X1[c])
        X1im[c] = np.imag(X1[c])

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


    # SIGMA


    def Sigma(coeff_matU, coeff_matX, ordre, V_slack):
        """
        coeff_matU: matriu de coeficients de tensió
        coeff_matX: matriu de coeficients de la tensió inversa conjugada
        ordre: profunditat seleccionada
        V_slack: tensions dels busos oscil·lants
        """

        V0 = V_slack  # tensió del bus oscil·lant de referència
        coeff_A = np.copy(coeff_matU)  # adaptar els coeficients per a la funció racional
        coeff_B= np.copy(coeff_matX)

        coeff_A[0] = 1
        for i in range(1, coeff_matU.shape[0]):
            coeff_A[i] = coeff_matU[i] - (V0 - 1) * coeff_A[i-1]
        coeff_B[0] = 1
        for i in range(1, coeff_matX.shape[0]):
            coeff_B[i] = coeff_matX[i] + (V0 - 1) * coeff_matX[i-1]

        sigmes = 0 + 0 * 1j

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

    # SIGMA
    Vx1 = np.copy(V1)
    Xx1 = np.copy(X1)
    Sig_re = np.real(Sigma(Vx1, Xx1, prof - 1, 1))
    Sig_im = np.imag(Sigma(Vx1, Xx1, prof - 1, 1))
    arrel = 0.25 + np.abs(Sig_re) - np.abs(Sig_im) ** 2
    Sigg = Sig_re + Sig_im * 1j
    sig_vec.append(Sigg)

    # print('Sigma real: ', Sig_re)
    # print('Sigma imaginary: ', Sig_im)
    # FI SIGMA


# SIG PLOT

a = []
b = []
c = []

x = np.linspace(-0.25, 1, 1000)
y = np.sqrt(0.25 + x)
a.append(x)
b.append(y)
c.append(-y)

plt.plot(np.real(sig_vec), np.imag(sig_vec), 'ro', markersize=2)
plt.plot(x, y)
plt.plot(x, -y)
plt.ylabel('Sigma im')
plt.xlabel('Sigma re')
plt.title('Gràfic Sigma')
plt.show()

# FI SIGMA