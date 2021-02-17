# embedding with Q(s), to check Sigma

import numpy as np

# DEFINITIONS 
uth = 1
zth = 0.01 + 0.1 * 1j
pref = 0.5
uref = 1
zvsc = 0.1j
W = 1 
Z0 = 0.8
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