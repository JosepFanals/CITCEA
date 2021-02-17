import numpy as np

P = 0.5
W = 1 * 1
Ytre = 0.01
Ytim = 0.1
Zo = 3
Yo = 1 / Zo  # although it takes different values, try this first

def f1(Vre, Vim, P, Ytre, Ytim, Yo):
    ff1 = P + Vre * (Ytre - Vre * Ytre + Vim * Ytim - Vre * Yo) + Vim * (Ytim - Vre * Ytim - Vim * Ytre - Vim * Yo)
    return ff1


def f2(Vre, Vim, W):
    ff2 = Vre ** 2 + Vim ** 2 - W
    return ff2


def f3(Vre, Vim, lam, c1, c2, c3, r):
    ff3 = (Vre - c1) ** 2 + (Vim - c2) ** 2 + (lam - c3) ** 2 - r ** 2
    return ff3


Vre0 = 1
Vim0 = 0.1

f10 = f1(Vre0, Vim0, P, Ytre, Ytim, Yo)
f20 = f2(Vre0, Vim0, W)

J = np.zeros((3, 3), dtype=float)
n_punts = 20000
n_iter = 5
r = 0.0005  # radius of circle
lam = 0.0
vec_Vre = []
vec_Vim = []
vec_lam = []
vec_error = []

# increments passats
AVre = 0
AVim = 0
Alam = 0

c1 = Vre0  # centre inicial de Vre
c2 = Vim0  # centre inicial de Vim
c3 = lam  # lambda inicial
Vre = Vre0  # valors de partida
Vim = Vim0  # valors de partid


for c in range(n_punts):
    # guardo valors del punt inicial
    Vre_i = Vre
    Vim_i = Vim
    lam_i = lam

    # etapa predictiva
    if c == 0:
        lam += r / 2  # és una bona inicialització, sabem que ha de créixer just al principi
    else:
        Vre += AVre
        Vim += AVim
        lam += Alam

    for k in range(n_iter):
        J[0, 0] = Ytre - 2 * Vre * Ytre + Vim * Ytim - 2 * Vre * Yo - Vim * Ytim
        J[0, 1] = Vre * Ytim + Ytim - Vre * Ytim - 2 * Vim * Ytre - 2 * Vim * Yo
        J[0, 2] = f10

        J[1, 0] = 2 * Vre
        J[1, 1] = 2 * Vim
        J[1, 2] = f20

        J[2, 0] = 2 * (Vre - c1)
        J[2, 1] = 2 * (Vim - c2)
        J[2, 2] = 2 * (lam - c3)

        h1 = f1(Vre, Vim, P, Ytre, Ytim, Yo) - (1 - lam) * f10
        h2 = f2(Vre, Vim, W) - (1 - lam) * f20
        h3 = f3(Vre, Vim, lam, c1, c2, c3, r)
        f = np.block([h1, h2, h3])
        J1 = np.linalg.inv(J)
        Ax = - np.dot(J1, f)

        Vre += Ax[0]
        Vim += Ax[1]
        lam += Ax[2]

    AVre = Vre - Vre_i
    AVim = Vim - Vim_i
    Alam = lam - lam_i
    c1 = Vre
    c2 = Vim
    c3 = lam
    vec_Vre.append(Vre)
    vec_Vim.append(Vim)
    vec_lam.append(lam)
    vec_error.append(max(abs(h1), abs(h2), abs(h3)))  # error màxim 


import matplotlib
import matplotlib.pyplot as plt

sub2 = plt.subplot(2, 1, 1) 
sub12 = plt.subplot(2, 1, 2) 
sub2.plot(vec_lam, vec_Vre)
sub12.plot(vec_lam, vec_Vim)
plt.show() 

print(vec_error)