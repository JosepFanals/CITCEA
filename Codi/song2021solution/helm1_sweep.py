# System 1 from song2021solution. 
# Values match with the rudimentary basic 


import numpy as np
import matplotlib.pyplot as plt

# DEFINITIONS 
uth = 1
zth = 0.01 + 0.1 * 1j
pref = 0.5
uref = 1
zvsc = 0.1j
W = 1 
Z0 = np.arange(0.2, 1.001, 0.001) 
n_test = int((1.001 - 0.2) / 0.001)
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


    V1f = sum(V1re) + sum(V1im) * 1j
    Io = V1f / Z0[kk]
    It = (1 - V1f) / zth
    Iv = Io - It
    Vvsc = V1f + Iv * zvsc

    Vvf[kk] = abs(Vvsc)

print('Vvf: ', Vvf)
plt.plot(Z0[:n_test], Vvf)
plt.ylabel('some numbers')
plt.show()