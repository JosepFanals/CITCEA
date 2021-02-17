#PGD code, trying to first construct the objects
import time
import pandas as pd
import numpy as np
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm

pd.options.display.precision = 2
pd.set_option('display.precision', 2)

df_b = pd.read_excel('data_100PQ.xlsx', sheet_name="buses")
df_l = pd.read_excel('data_100PQ.xlsx', sheet_name="lines")

n_b = 0
n_pq = 0
n_pv = 0
pq = []
pv = []
pq0 = []  # store pq buses indices relative to its own
pv0 = []  # store pv buses indices relative to its own
d_pq = {}  # dict of pq
d_pv = {}  # dict of pv
for i in range(len(df_b)):
    if df_b.iloc[i, 4] == "slack":  # index 0 is reserved for the slack bus
        pass

    elif df_b.iloc[i, 4] == "PQ":
        pq0.append(n_pq)
        d_pq[df_b.iloc[i, 0]] = n_pq
        n_b += 1
        n_pq += 1
        pq.append(df_b.iloc[i, 0] - 1)
        
    elif df_b.iloc[i, 4] == "PV":
        pv0.append(n_pv)
        d_pv[df_b.iloc[i, 0]] = n_pv
        n_b += 1
        n_pv += 1
        pv.append(df_b.iloc[i, 0] - 1)

n_l = len(df_l)  # number of lines

V0 = df_b.iloc[0, 3]  # the slack is always positioned in the first row
I0_pq = np.zeros(n_pq, dtype=complex)
I0_pv = np.zeros(n_pv, dtype=complex)
Y = np.zeros((n_b, n_b), dtype=complex)  # I will build it with block matrices
Y11 = np.zeros((n_pq, n_pq), dtype=complex)  # pq pq
Y12 = np.zeros((n_pq, n_pv), dtype=complex)  # pq pv
Y21 = np.zeros((n_pv, n_pq), dtype=complex)  # pv pq
Y22 = np.zeros((n_pv, n_pv), dtype=complex)  # pv pv

for i in range(n_l):
    Ys = 1 / (df_l.iloc[i, 2] + 1j * df_l.iloc[i, 3])  # series element
    Ysh = df_l.iloc[i, 4] + 1j * df_l.iloc[i, 5]  # shunt element
    t = df_l.iloc[i, 6] * np.cos(df_l.iloc[i, 7]) + 1j * df_l.iloc[i, 6] * np.sin(df_l.iloc[i, 7])  # tap as a complex number

    a = df_l.iloc[i, 0]
    b = df_l.iloc[i, 1]

    if a == 0:
        if b - 1 in pq:
            I0_pq[d_pq[b]] += V0 * Ys / t
            Y11[d_pq[b], d_pq[b]] += Ys + Ysh
        if b - 1 in pv:
            I0_pv[d_pv[b]] += V0 * Ys / t
            Y22[d_pv[b], d_pv[b]] += Ys + Ysh

    elif b == 0:
        if a - 1 in pq:
            I0_pq[d_pq[a]] += V0 * Ys / np.conj(t)
            Y11[d_pq[a], d_pq[a]] += (Ys + Ysh) / (t * np.conj(t))
        if a - 1 in pv:
            I0_pv[d_pv[a]] += V0 * Ys / np.conj(t)
            Y22[d_pv[a], d_pv[a]] += (Ys + Ysh) / (t * np.conj(t))

    else:
        if a - 1 in pq and b - 1 in pq:
            Y11[d_pq[a], d_pq[a]] += (Ys + Ysh) / (t * np.conj(t))
            Y11[d_pq[b], d_pq[b]] += Ys + Ysh
            Y11[d_pq[a], d_pq[b]] += - Ys / np.conj(t)
            Y11[d_pq[b], d_pq[a]] += - Ys / t
        
        if a - 1 in pq and b - 1 in pv:
            Y11[d_pq[a], d_pq[a]] += (Ys + Ysh) / (t * np.conj(t))
            Y22[d_pv[b], d_pv[b]] += Ys + Ysh
            Y12[d_pq[a], d_pv[b]] += - Ys / np.conj(t)
            Y21[d_pv[b], d_pq[a]] += - Ys / t

        if a - 1 in pv and b - 1 in pq:
            Y22[d_pv[a], d_pv[a]] += (Ys + Ysh) / (t * np.conj(t))
            Y11[d_pq[b], d_pq[b]] += Ys + Ysh
            Y21[d_pv[a], d_pq[b]] += - Ys / np.conj(t)
            Y12[d_pq[b], d_pv[a]] += - Ys / t

        if a - 1 in pv and b - 1 in pv:
            Y22[d_pv[a], d_pv[a]] += (Ys + Ysh) / (t * np.conj(t))
            Y22[d_pv[b], d_pv[b]] += Ys + Ysh
            Y22[d_pv[a], d_pv[b]] += - Ys / np.conj(t)
            Y22[d_pv[b], d_pv[a]] += - Ys / t


for i in range(len(df_b)):  # add shunts connected directly to the bus
    a = df_b.iloc[i, 0]
    if a - 1 in pq:
        # print(d_pq[a])
        Y11[d_pq[a], d_pq[a]] += df_b.iloc[i, 5] + 1j * df_b.iloc[i, 6]
    elif a - 1 in pv:
        # print(d_pv[a])
        Y22[d_pv[a], d_pv[a]] += df_b.iloc[i, 5] + 1j * df_b.iloc[i, 6]


Y = np.block([[Y11, Y12], [Y21, Y22]])
Yinv = np.linalg.inv(Y)
Ydf = pd.DataFrame(Y)

V_mod = np.zeros(n_pv, dtype=float)
P_pq = np.zeros(n_pq, dtype=float)
P_pv = np.zeros(n_pv, dtype=float)
Q_pq = np.zeros(n_pq, dtype=float)
for i in range(len(df_b)):
    if df_b.iloc[i, 4] == "PV":
        V_mod[d_pv[df_b.iloc[i, 0]]] = df_b.iloc[i, 3]
        P_pv[d_pv[df_b.iloc[i, 0]]] = df_b.iloc[i, 1]
    elif df_b.iloc[i, 4] == "PQ":
        Q_pq[d_pq[df_b.iloc[i, 0]]] = df_b.iloc[i, 2]
        P_pq[d_pq[df_b.iloc[i, 0]]] = df_b.iloc[i, 1]

# print(Ydf)
# print(V_mod)
# print(P_pq)
# print(P_pv)
# print(Q_pq)

Kk1 = np.zeros(102, dtype=complex)  # power amplitude, in this case, it will be reactive
Kk1[0] = 1 * 1j  # put a 1 because only one capacitor at a time
Pp1 = np.ones(102)  # possible positions of the capacitor
Pp1[0] = 0  # bus1, we do not consider the possibility to include the capacitor here
Pp1[1] = 0  # bus2, we do not consider the possibility to include the capacitor here
Qq1 = np.arange(0,1.001,0.001)  # scale the powers in a discrete form
Tpq1 = np.multiply.outer(Kk1, np.multiply.outer(Pp1, Qq1))  # tensor for the capacitor
# print(Tpq1)

Kk2 = P_pq + Q_pq * 1j  # original load
Pp2 = np.ones(102)  # positions of standard the loads
Qq2 = np.ones(1001)  # always multiply by a factor of 1, the original loads do not change
Tpq2 = np.multiply.outer(Kk2, np.multiply.outer(Pp2, Qq2))  # tensor for the normal loads
# print(Tpq2)

TS = Tpq1 + Tpq2  # full tensor for the power
TSc = np.conj(TS)  # conjugated power tensor

Kkv = np.ones(102, dtype=complex)  # amplitude vector
Ppv = np.ones(102)  # position vector
Qqv = np.ones(1001)  # scaling vector
TVv = np.multiply.outer(Kkv, np.multiply.outer(Ppv, Qqv))
TVc = np.conj(TVv)
# print(TVc)

# it is not a good idea to compute the full tensor!! we work with the vectors that form it individually!!
VVk1 = np.conj(Kkv)
VVp1 = np.conj(Ppv)
VVq1 = np.conj(Qqv)

VVr = [VVk1, VVp1, VVq1]

# -----------------BEGIN OUTER LOOP NUMBER 1-----------------

SSk1 = np.conj(Kk1)
SSp1 = np.conj(Pp1)
SSq1 = np.conj(Qq1)
SSk2 = np.conj(Kk2)
SSp2 = np.conj(Pp2)
SSq2 = np.conj(Qq2)

SSr = [[SSk1, SSp1, SSq1], [SSk2, SSp2, SSq2]]

CCk1 = SSk1
CCp1 = SSp1
CCq1 = SSq1
CCk2 = SSk2
CCp2 = SSp2
CCq2 = SSq2

CCr = [[CCk1, CCp1, CCq1], [CCk2, CCp2, CCq2]]

# IIk1 = np.random.rand(102)
IIk1 = VVk1  # try this
IIp1 = np.random.rand(102)
IIq1 = np.random.rand(1001)

# inner loop would start here. This would be the loop on Gamma
for k in range(10):  # we could choose more iterations
    # compute IIk1
    prodC1 = np.dot(IIp1, CCp1) * np.dot(IIq1, CCq1)
    prodC2 = np.dot(IIp1, CCp2) * np.dot(IIq1, CCq2)
    RHS = prodC1 * CCk1 + prodC2 * CCk2

    prodL1 = np.dot(IIp1, VVp1 * IIp1) * np.dot(IIq1, VVq1 * IIq1)
    LHS = prodL1 * VVk1
    IIk1 = RHS / LHS

    # compute IIp1
    prodC1 = np.dot(IIk1, CCk1) * np.dot(IIq1, CCq1)
    prodC2 = np.dot(IIk1, CCk2) * np.dot(IIq1, CCq2)
    RHS = prodC1 * CCp1 + prodC2 * CCp2

    prodL1 = np.dot(IIk1, VVk1 * IIk1) * np.dot(IIq1, VVq1 * IIq1)
    LHS = prodL1 * VVp1
    IIp1 = RHS / LHS

    # compute IIq1
    prodC1 = np.dot(IIk1, CCk1) * np.dot(IIp1, CCp1)
    prodC2 = np.dot(IIk1, CCk2) * np.dot(IIp1, CCp2)
    RHS = prodC1 * CCq1 + prodC2 * CCq2

    prodL1 = np.dot(IIk1, VVk1 * IIk1) * np.dot(IIp1, VVp1 * IIp1)
    LHS = prodL1 * VVq1
    IIq1 = RHS / LHS

# another iteration of the intermediate loop
CCk3 = - VVk1 * IIk1  # negative sign because S* - V*I
CCp3 = - VVp1 * IIp1
CCq3 = - VVq1 * IIq1

IIk2 = np.random.rand(102) 
IIp2 = np.random.rand(102)
IIq2 = np.random.rand(1001)

for k in range(10):  # we could choose more iterations
    # compute IIk2
    prodC1 = np.dot(IIp2, CCp1) * np.dot(IIq2, CCq1)
    prodC2 = np.dot(IIp2, CCp2) * np.dot(IIq2, CCq2)
    prodC3 = np.dot(IIp2, CCp3) * np.dot(IIq2, CCq3)
    RHS = prodC1 * CCk1 + prodC2 * CCk2 + prodC3 * CCk3

    prodL1 = np.dot(IIp2, VVp1 * IIp2) * np.dot(IIq2, VVq1 * IIq2)
    LHS = prodL1 * VVk1
    IIk2 = RHS / LHS

    # compute IIp2
    prodC1 = np.dot(IIk2, CCk1) * np.dot(IIq2, CCq1)
    prodC2 = np.dot(IIk2, CCk2) * np.dot(IIq2, CCq2)
    prodC3 = np.dot(IIk2, CCk3) * np.dot(IIq2, CCq3)
    RHS = prodC1 * CCp1 + prodC2 * CCp2 + prodC3 * CCp3

    prodL1 = np.dot(IIk2, VVk1 * IIk2) * np.dot(IIq2, VVq1 * IIq2)
    LHS = prodL1 * VVp1
    IIp2 = RHS / LHS

    # compute IIq2
    prodC1 = np.dot(IIk2, CCk1) * np.dot(IIp2, CCp1)
    prodC2 = np.dot(IIk2, CCk2) * np.dot(IIp2, CCp2)
    prodC3 = np.dot(IIk2, CCk3) * np.dot(IIp2, CCp3)
    RHS = prodC1 * CCq1 + prodC2 * CCq2 + prodC3 * CCq3

    prodL1 = np.dot(IIk2, VVk1 * IIk2) * np.dot(IIp2, VVp1 * IIp2)
    LHS = prodL1 * VVq1
    IIq2 = RHS / LHS

# another iteration of the intermediate loop
CCk4 = - VVk1 * IIk2  # negative sign because S* - V*I
CCp4 = - VVp1 * IIp2
CCq4 = - VVq1 * IIq2

IIk3 = np.random.rand(102) 
IIp3 = np.random.rand(102)
IIq3 = np.random.rand(1001)

for k in range(10):  # we could choose more iterations
    # compute IIk3
    prodC1 = np.dot(IIp3, CCp1) * np.dot(IIq3, CCq1)
    prodC2 = np.dot(IIp3, CCp2) * np.dot(IIq3, CCq2)
    prodC3 = np.dot(IIp3, CCp3) * np.dot(IIq3, CCq3)
    prodC4 = np.dot(IIp3, CCp4) * np.dot(IIq3, CCq4)
    RHS = prodC1 * CCk1 + prodC2 * CCk2 + prodC3 * CCk3 + prodC4 * CCk4

    prodL1 = np.dot(IIp3, VVp1 * IIp3) * np.dot(IIq3, VVq1 * IIq3)
    LHS = prodL1 * VVk1
    IIk3 = RHS / LHS

    # compute IIp3
    prodC1 = np.dot(IIk3, CCk1) * np.dot(IIq3, CCq1)
    prodC2 = np.dot(IIk3, CCk2) * np.dot(IIq3, CCq2)
    prodC3 = np.dot(IIk3, CCk3) * np.dot(IIq3, CCq3)
    prodC4 = np.dot(IIk3, CCk4) * np.dot(IIq3, CCq4)
    RHS = prodC1 * CCp1 + prodC2 * CCp2 + prodC3 * CCp3 + prodC4 * CCp4

    prodL1 = np.dot(IIk3, VVk1 * IIk3) * np.dot(IIq3, VVq1 * IIq3)
    LHS = prodL1 * VVp1
    IIp3 = RHS / LHS

    # compute IIq3
    prodC1 = np.dot(IIk3, CCk1) * np.dot(IIp3, CCp1)
    prodC2 = np.dot(IIk3, CCk2) * np.dot(IIp3, CCp2)
    prodC3 = np.dot(IIk3, CCk3) * np.dot(IIp3, CCp3)
    prodC4 = np.dot(IIk3, CCk4) * np.dot(IIp3, CCp4)
    RHS = prodC1 * CCq1 + prodC2 * CCq2 + prodC3 * CCq3 + prodC4 * CCq4

    prodL1 = np.dot(IIk3, VVk1 * IIk3) * np.dot(IIp3, VVp1 * IIp3)
    LHS = prodL1 * VVq1
    IIq3 = RHS / LHS

# another iteration of the intermediate loop
CCk5 = - VVk1 * IIk3  # negative sign because S* - V*I
CCp5 = - VVp1 * IIp3
CCq5 = - VVq1 * IIq3

IIk4 = np.random.rand(102) 
IIp4 = np.random.rand(102)
IIq4 = np.random.rand(1001)

for k in range(10):  # we could choose more iterations
    # compute IIk3
    prodC1 = np.dot(IIp4, CCp1) * np.dot(IIq4, CCq1)
    prodC2 = np.dot(IIp4, CCp2) * np.dot(IIq4, CCq2)
    prodC3 = np.dot(IIp4, CCp3) * np.dot(IIq4, CCq3)
    prodC4 = np.dot(IIp4, CCp4) * np.dot(IIq4, CCq4)
    prodC5 = np.dot(IIp4, CCp5) * np.dot(IIq4, CCq5)
    RHS = prodC1 * CCk1 + prodC2 * CCk2 + prodC3 * CCk3 + prodC4 * CCk4 + prodC5 * CCk5

    prodL1 = np.dot(IIp4, VVp1 * IIp4) * np.dot(IIq4, VVq1 * IIq4)
    LHS = prodL1 * VVk1
    IIk4 = RHS / LHS

    # compute IIp3
    prodC1 = np.dot(IIk4, CCk1) * np.dot(IIq4, CCq1)
    prodC2 = np.dot(IIk4, CCk2) * np.dot(IIq4, CCq2)
    prodC3 = np.dot(IIk4, CCk3) * np.dot(IIq4, CCq3)
    prodC4 = np.dot(IIk4, CCk4) * np.dot(IIq4, CCq4)
    prodC5 = np.dot(IIk4, CCk5) * np.dot(IIq4, CCq5)
    RHS = prodC1 * CCp1 + prodC2 * CCp2 + prodC3 * CCp3 + prodC4 * CCp4 + prodC5 * CCp5

    prodL1 = np.dot(IIk4, VVk1 * IIk4) * np.dot(IIq4, VVq1 * IIq4)
    LHS = prodL1 * VVp1
    IIp4 = RHS / LHS

    # compute IIq3
    prodC1 = np.dot(IIk4, CCk1) * np.dot(IIp4, CCp1)
    prodC2 = np.dot(IIk4, CCk2) * np.dot(IIp4, CCp2)
    prodC3 = np.dot(IIk4, CCk3) * np.dot(IIp4, CCp3)
    prodC4 = np.dot(IIk4, CCk4) * np.dot(IIp4, CCp4)
    prodC5 = np.dot(IIk4, CCk5) * np.dot(IIp4, CCp5)
    RHS = prodC1 * CCq1 + prodC2 * CCq2 + prodC3 * CCq3 + prodC4 * CCq4 + prodC5 * CCq5

    prodL1 = np.dot(IIk4, VVk1 * IIk4) * np.dot(IIp4, VVp1 * IIp4)
    LHS = prodL1 * VVq1
    IIq4 = RHS / LHS

# now I have done a lot of inner loops and about 4 intermediate loops.
# I can now proceed to update the voltage and iterate on gamma
# so this is another outer loop:

VVk1 = np.dot(Yinv, IIk1)
VVk2 = np.dot(Yinv, IIk2)
VVk3 = np.dot(Yinv, IIk3)
VVk4 = np.dot(Yinv, IIk4)

# -----------------END OUTER LOOP NUMBER 1-----------------

# -----------------BEGIN OUTER LOOP NUMBER 2-----------------

VVk1 = np.conj(VVk1)
VVk2 = np.conj(VVk2)
VVk3 = np.conj(VVk3)
VVk4 = np.conj(VVk4)

VVp1 = np.conj(IIp1)
VVp2 = np.conj(IIp2)
VVp3 = np.conj(IIp3)
VVp4 = np.conj(IIp4)

VVq1 = np.conj(IIq1)
VVq2 = np.conj(IIq2)
VVq3 = np.conj(IIq3)
VVq4 = np.conj(IIq4)

# VVr = [VVk1, VVp1, VVq1]

SSk1 = np.conj(Kk1)
SSp1 = np.conj(Pp1)
SSq1 = np.conj(Qq1)
SSk2 = np.conj(Kk2)
SSp2 = np.conj(Pp2)
SSq2 = np.conj(Qq2)

SSr = [[SSk1, SSp1, SSq1], [SSk2, SSp2, SSq2]]

CCk1 = SSk1
CCp1 = SSp1
CCq1 = SSq1
CCk2 = SSk2
CCp2 = SSp2
CCq2 = SSq2

CCr = [[CCk1, CCp1, CCq1], [CCk2, CCp2, CCq2]]

# we can start the currents with the previos values, probably it will be good
# IIk1 = VVk1  # try this
# IIp1 = np.random.rand(102)
# IIq1 = np.random.rand(1001)

# inner loop would start here. This would be the loop on Gamma
for k in range(10):  # we could choose more iterations
    # compute IIk1
    prodC1 = np.dot(IIp1, CCp1) * np.dot(IIq1, CCq1)
    prodC2 = np.dot(IIp1, CCp2) * np.dot(IIq1, CCq2)
    RHS = prodC1 * CCk1 + prodC2 * CCk2

    prodL1 = np.dot(IIp1, VVp1 * IIp1) * np.dot(IIq1, VVq1 * IIq1)
    prodL2 = np.dot(IIp1, VVp2 * IIp1) * np.dot(IIq1, VVq2 * IIq1)
    prodL3 = np.dot(IIp1, VVp3 * IIp1) * np.dot(IIq1, VVq3 * IIq1)
    prodL4 = np.dot(IIp1, VVp4 * IIp1) * np.dot(IIq1, VVq4 * IIq1)
    LHS = prodL1 * VVk1 + prodL2 * VVk2 + prodL3 * VVk3 + prodL4 * VVk4
    IIk1 = RHS / LHS

    # compute IIp1
    prodC1 = np.dot(IIk1, CCk1) * np.dot(IIq1, CCq1)
    prodC2 = np.dot(IIk1, CCk2) * np.dot(IIq1, CCq2)
    RHS = prodC1 * CCp1 + prodC2 * CCp2

    prodL1 = np.dot(IIk1, VVk1 * IIk1) * np.dot(IIq1, VVq1 * IIq1)
    prodL2 = np.dot(IIk1, VVk2 * IIk1) * np.dot(IIq1, VVq2 * IIq1)
    prodL3 = np.dot(IIk1, VVk3 * IIk1) * np.dot(IIq1, VVq3 * IIq1)
    prodL4 = np.dot(IIk1, VVk4 * IIk1) * np.dot(IIq1, VVq4 * IIq1)
    LHS = prodL1 * VVp1 + prodL2 * VVp2 + prodL3 * VVp3 + prodL4 * VVp4
    IIp1 = RHS / LHS

    # compute IIq1
    prodC1 = np.dot(IIk1, CCk1) * np.dot(IIp1, CCp1)
    prodC2 = np.dot(IIk1, CCk2) * np.dot(IIp1, CCp2)
    RHS = prodC1 * CCq1 + prodC2 * CCq2

    prodL1 = np.dot(IIk1, VVk1 * IIk1) * np.dot(IIp1, VVp1 * IIp1)
    prodL2 = np.dot(IIk1, VVk2 * IIk1) * np.dot(IIp1, VVp2 * IIp1)
    prodL3 = np.dot(IIk1, VVk3 * IIk1) * np.dot(IIp1, VVp3 * IIp1)
    prodL4 = np.dot(IIk1, VVk4 * IIk1) * np.dot(IIp1, VVp3 * IIp1)
    LHS = prodL1 * VVq1 + prodL2 * VVq2 + prodL3 * VVq3 + prodL4 * VVq4
    IIq1 = RHS / LHS


# another iteration of the intermediate loop
CCk3 = - VVk1 * IIk1  # negative sign because S* - V*I
CCp3 = - VVp1 * IIp1
CCq3 = - VVq1 * IIq1

CCk4 = - VVk2 * IIk1
CCp4 = - VVp2 * IIp1
CCq4 = - VVq2 * IIq1

CCk5 = - VVk3 * IIk1
CCp5 = - VVp3 * IIp1
CCq5 = - VVq3 * IIq1

CCk6 = - VVk4 * IIk1
CCp6 = - VVp4 * IIp1
CCq6 = - VVq4 * IIq1

# IIk2 = np.random.rand(102) 
# IIp2 = np.random.rand(102)
# IIq2 = np.random.rand(1001)

for k in range(10):  # we could choose more iterations
    # compute IIk2
    prodC1 = np.dot(IIp2, CCp1) * np.dot(IIq2, CCq1)
    prodC2 = np.dot(IIp2, CCp2) * np.dot(IIq2, CCq2)
    prodC3 = np.dot(IIp2, CCp3) * np.dot(IIq2, CCq3)
    prodC4 = np.dot(IIp2, CCp4) * np.dot(IIq2, CCq4)
    prodC5 = np.dot(IIp2, CCp5) * np.dot(IIq2, CCq5)
    prodC6 = np.dot(IIp2, CCp6) * np.dot(IIq2, CCq6)
    RHS = prodC1 * CCk1 + prodC2 * CCk2 + prodC3 * CCk3 + prodC4 * CCk4 + prodC5 * CCk5 + prodC6 * CCk6

    prodL1 = np.dot(IIp2, VVp1 * IIp2) * np.dot(IIq2, VVq1 * IIq2)
    prodL2 = np.dot(IIp2, VVp2 * IIp2) * np.dot(IIq2, VVq2 * IIq2)
    prodL3 = np.dot(IIp2, VVp3 * IIp2) * np.dot(IIq2, VVq3 * IIq2)
    prodL4 = np.dot(IIp2, VVp4 * IIp2) * np.dot(IIq2, VVq4 * IIq2)
    LHS = prodL1 * VVk1 + prodL2 * VVk2 + prodL3 * VVk3 + prodL4 * VVk4

    IIk2 = RHS / LHS

    # compute IIp2
    prodC1 = np.dot(IIk2, CCk1) * np.dot(IIq2, CCq1)
    prodC2 = np.dot(IIk2, CCk2) * np.dot(IIq2, CCq2)
    prodC3 = np.dot(IIk2, CCk3) * np.dot(IIq2, CCq3)
    prodC4 = np.dot(IIk2, CCk4) * np.dot(IIq2, CCq4)
    prodC5 = np.dot(IIk2, CCk5) * np.dot(IIq2, CCq5)
    prodC6 = np.dot(IIk2, CCk6) * np.dot(IIq2, CCq6)
    RHS = prodC1 * CCp1 + prodC2 * CCp2 + prodC3 * CCp3 + prodC4 * CCp4 + prodC5 * CCp5 + prodC6 * CCp6

    prodL1 = np.dot(IIk2, VVk1 * IIk2) * np.dot(IIq2, VVq1 * IIq2)
    prodL2 = np.dot(IIk2, VVk2 * IIk2) * np.dot(IIq2, VVq2 * IIq2)
    prodL3 = np.dot(IIk2, VVk3 * IIk2) * np.dot(IIq2, VVq3 * IIq2)
    prodL4 = np.dot(IIk2, VVk4 * IIk2) * np.dot(IIq2, VVq4 * IIq2)
    LHS = prodL1 * VVp1 + prodL2 * VVp2 + prodL3 * VVp3 + prodL4 * VVp4

    IIp2 = RHS / LHS

    # compute IIq2
    prodC1 = np.dot(IIk2, CCk1) * np.dot(IIp2, CCp1)
    prodC2 = np.dot(IIk2, CCk2) * np.dot(IIp2, CCp2)
    prodC3 = np.dot(IIk2, CCk3) * np.dot(IIp2, CCp3)
    prodC4 = np.dot(IIk2, CCk4) * np.dot(IIp2, CCp4)
    prodC5 = np.dot(IIk2, CCk5) * np.dot(IIp2, CCp5)
    prodC6 = np.dot(IIk2, CCk6) * np.dot(IIp2, CCp6)
    RHS = prodC1 * CCq1 + prodC2 * CCq2 + prodC3 * CCq3 + prodC4 * CCq4 + prodC5 * CCq5 + prodC6 * CCq6

    prodL1 = np.dot(IIk2, VVk1 * IIk2) * np.dot(IIp2, VVp1 * IIp2)
    prodL2 = np.dot(IIk2, VVk2 * IIk2) * np.dot(IIp2, VVp2 * IIp2)
    prodL3 = np.dot(IIk2, VVk3 * IIk2) * np.dot(IIp2, VVp3 * IIp2)
    prodL4 = np.dot(IIk2, VVk4 * IIk2) * np.dot(IIp2, VVp3 * IIp2)
    LHS = prodL1 * VVq1 + prodL2 * VVq2 + prodL3 * VVq3 + prodL4 * VVq4

    IIq2 = RHS / LHS

# another iteration of the intermediate loop
CCk7 = - VVk1 * IIk2  # negative sign because S* - V*I
CCp7 = - VVp1 * IIp2
CCq7 = - VVq1 * IIq2

CCk8 = - VVk2 * IIk2
CCp8 = - VVp2 * IIp2
CCq8 = - VVq2 * IIq2

CCk9 = - VVk3 * IIk2
CCp9 = - VVp3 * IIp2
CCq9 = - VVq3 * IIq2

CCk10 = - VVk4 * IIk2
CCp10 = - VVp4 * IIp2
CCq10 = - VVq4 * IIq2

for k in range(10):  # we could choose more iterations
    # compute IIk3
    prodC1 = np.dot(IIp3, CCp1) * np.dot(IIq3, CCq1)
    prodC2 = np.dot(IIp3, CCp2) * np.dot(IIq3, CCq2)
    prodC3 = np.dot(IIp3, CCp3) * np.dot(IIq3, CCq3)
    prodC4 = np.dot(IIp3, CCp4) * np.dot(IIq3, CCq4)
    prodC5 = np.dot(IIp3, CCp5) * np.dot(IIq3, CCq5)
    prodC6 = np.dot(IIp3, CCp6) * np.dot(IIq3, CCq6)
    prodC7 = np.dot(IIp3, CCp7) * np.dot(IIq3, CCq7)
    prodC8 = np.dot(IIp3, CCp8) * np.dot(IIq3, CCq8)
    prodC9 = np.dot(IIp3, CCp9) * np.dot(IIq3, CCq9)
    prodC10 = np.dot(IIp3, CCp10) * np.dot(IIq3, CCq10)
    RHS = prodC1 * CCk1 + prodC2 * CCk2 + prodC3 * CCk3 + prodC4 * CCk4 + prodC5 * CCk5 + prodC6 * CCk6 + prodC7 * CCk7 + prodC8 * CCk8 + prodC9 * CCk9 + prodC10 * CCk10

    prodL1 = np.dot(IIp3, VVp1 * IIp3) * np.dot(IIq3, VVq1 * IIq3)
    prodL2 = np.dot(IIp3, VVp2 * IIp3) * np.dot(IIq3, VVq2 * IIq3)
    prodL3 = np.dot(IIp3, VVp3 * IIp3) * np.dot(IIq3, VVq3 * IIq3)
    prodL4 = np.dot(IIp3, VVp4 * IIp3) * np.dot(IIq3, VVq4 * IIq3)
    LHS = prodL1 * VVk1 + prodL2 * VVk2 + prodL3 * VVk3 + prodL4 * VVk4

    IIk3 = RHS / LHS

    # compute IIp3
    prodC1 = np.dot(IIk3, CCk1) * np.dot(IIq3, CCq1)
    prodC2 = np.dot(IIk3, CCk2) * np.dot(IIq3, CCq2)
    prodC3 = np.dot(IIk3, CCk3) * np.dot(IIq3, CCq3)
    prodC4 = np.dot(IIk3, CCk4) * np.dot(IIq3, CCq4)
    prodC5 = np.dot(IIk3, CCk5) * np.dot(IIq3, CCq5)
    prodC6 = np.dot(IIk3, CCk6) * np.dot(IIq3, CCq6)
    prodC7 = np.dot(IIk3, CCk7) * np.dot(IIq3, CCq7)
    prodC8 = np.dot(IIk3, CCk8) * np.dot(IIq3, CCq8)
    prodC9 = np.dot(IIk3, CCk9) * np.dot(IIq3, CCq9)
    prodC10 = np.dot(IIk3, CCk10) * np.dot(IIq3, CCq10)
    RHS = prodC1 * CCp1 + prodC2 * CCp2 + prodC3 * CCp3 + prodC4 * CCp4 + prodC5 * CCp5 + prodC6 * CCp6 + prodC7 * CCp7 + prodC8 * CCp8 + prodC9 * CCp9 + prodC10 * CCp10

    prodL1 = np.dot(IIk3, VVk1 * IIk3) * np.dot(IIq3, VVq1 * IIq3)
    prodL2 = np.dot(IIk3, VVk2 * IIk3) * np.dot(IIq3, VVq2 * IIq3)
    prodL3 = np.dot(IIk3, VVk3 * IIk3) * np.dot(IIq3, VVq3 * IIq3)
    prodL4 = np.dot(IIk3, VVk4 * IIk3) * np.dot(IIq3, VVq4 * IIq3)
    LHS = prodL1 * VVp1 + prodL2 * VVp2 + prodL3 * VVp3 + prodL4 * VVp4

    IIp3 = RHS / LHS

    # compute IIq3
    prodC1 = np.dot(IIk3, CCk1) * np.dot(IIp3, CCp1)
    prodC2 = np.dot(IIk3, CCk2) * np.dot(IIp3, CCp2)
    prodC3 = np.dot(IIk3, CCk3) * np.dot(IIp3, CCp3)
    prodC4 = np.dot(IIk3, CCk4) * np.dot(IIp3, CCp4)
    prodC5 = np.dot(IIk3, CCk5) * np.dot(IIp3, CCp5)
    prodC6 = np.dot(IIk3, CCk6) * np.dot(IIp3, CCp6)
    prodC7 = np.dot(IIk3, CCk7) * np.dot(IIp3, CCp7)
    prodC8 = np.dot(IIk3, CCk8) * np.dot(IIp3, CCp8)
    prodC9 = np.dot(IIk3, CCk9) * np.dot(IIp3, CCp9)
    prodC10 = np.dot(IIk3, CCk10) * np.dot(IIp3, CCp10)
    RHS = prodC1 * CCq1 + prodC2 * CCq2 + prodC3 * CCq3 + prodC4 * CCq4 + prodC5 * CCq5 + prodC6 * CCq6 + prodC7 * CCq7 + prodC8 * CCq8 + prodC9 * CCq9 + prodC10 * CCq10

    prodL1 = np.dot(IIk3, VVk1 * IIk3) * np.dot(IIp3, VVp1 * IIp3)
    prodL2 = np.dot(IIk3, VVk2 * IIk3) * np.dot(IIp3, VVp2 * IIp3)
    prodL3 = np.dot(IIk3, VVk3 * IIk3) * np.dot(IIp3, VVp3 * IIp3)
    prodL4 = np.dot(IIk3, VVk4 * IIk3) * np.dot(IIp3, VVp3 * IIp3)
    LHS = prodL1 * VVq1 + prodL2 * VVq2 + prodL3 * VVq3 + prodL4 * VVq4

    IIq3 = RHS / LHS

# another iteration of the intermediate loop
CCk11 = - VVk1 * IIk3  # negative sign because S* - V*I
CCp11 = - VVp1 * IIp3
CCq11 = - VVq1 * IIq3

CCk12 = - VVk2 * IIk3
CCp12 = - VVp2 * IIp3
CCq12 = - VVq2 * IIq3

CCk13 = - VVk3 * IIk3
CCp13 = - VVp3 * IIp3
CCq13 = - VVq3 * IIq3

CCk14 = - VVk4 * IIk3
CCp14 = - VVp4 * IIp3
CCq14 = - VVq4 * IIq3

for k in range(10):  # we could choose more iterations
    # compute IIk4
    prodC1 = np.dot(IIp4, CCp1) * np.dot(IIq4, CCq1)
    prodC2 = np.dot(IIp4, CCp2) * np.dot(IIq4, CCq2)
    prodC3 = np.dot(IIp4, CCp3) * np.dot(IIq4, CCq3)
    prodC4 = np.dot(IIp4, CCp4) * np.dot(IIq4, CCq4)
    prodC5 = np.dot(IIp4, CCp5) * np.dot(IIq4, CCq5)
    prodC6 = np.dot(IIp4, CCp6) * np.dot(IIq4, CCq6)
    prodC7 = np.dot(IIp4, CCp7) * np.dot(IIq4, CCq7)
    prodC8 = np.dot(IIp4, CCp8) * np.dot(IIq4, CCq8)
    prodC9 = np.dot(IIp4, CCp9) * np.dot(IIq4, CCq9)
    prodC10 = np.dot(IIp4, CCp10) * np.dot(IIq4, CCq10)
    prodC11 = np.dot(IIp4, CCp11) * np.dot(IIq4, CCq11)
    prodC12 = np.dot(IIp4, CCp12) * np.dot(IIq4, CCq12)
    prodC13 = np.dot(IIp4, CCp13) * np.dot(IIq4, CCq13)
    prodC14 = np.dot(IIp4, CCp14) * np.dot(IIq4, CCq14)
    RHS = prodC1 * CCk1 + prodC2 * CCk2 + prodC3 * CCk3 + prodC4 * CCk4 + prodC5 * CCk5 + prodC6 * CCk6 + prodC7 * CCk7 + prodC8 * CCk8 + prodC9 * CCk9 + prodC10 * CCk10 + prodC11 * CCk11 + prodC12 * CCk12 + prodC13 * CCk13 + prodC14 * CCk14

    prodL1 = np.dot(IIp4, VVp1 * IIp4) * np.dot(IIq4, VVq1 * IIq4)
    prodL2 = np.dot(IIp4, VVp2 * IIp4) * np.dot(IIq4, VVq2 * IIq4)
    prodL3 = np.dot(IIp4, VVp3 * IIp4) * np.dot(IIq4, VVq3 * IIq4)
    prodL4 = np.dot(IIp4, VVp4 * IIp4) * np.dot(IIq4, VVq4 * IIq4)
    LHS = prodL1 * VVk1 + prodL2 * VVk2 + prodL3 * VVk3 + prodL4 * VVk4

    IIk4 = RHS / LHS

    # compute IIp4
    prodC1 = np.dot(IIk4, CCk1) * np.dot(IIq4, CCq1)
    prodC2 = np.dot(IIk4, CCk2) * np.dot(IIq4, CCq2)
    prodC3 = np.dot(IIk4, CCk3) * np.dot(IIq4, CCq3)
    prodC4 = np.dot(IIk4, CCk4) * np.dot(IIq4, CCq4)
    prodC5 = np.dot(IIk4, CCk5) * np.dot(IIq4, CCq5)
    prodC6 = np.dot(IIk4, CCk6) * np.dot(IIq4, CCq6)
    prodC7 = np.dot(IIk4, CCk7) * np.dot(IIq4, CCq7)
    prodC8 = np.dot(IIk4, CCk8) * np.dot(IIq4, CCq8)
    prodC9 = np.dot(IIk4, CCk9) * np.dot(IIq4, CCq9)
    prodC10 = np.dot(IIk4, CCk10) * np.dot(IIq4, CCq10)
    prodC11 = np.dot(IIk4, CCk11) * np.dot(IIq4, CCq11)
    prodC12 = np.dot(IIk4, CCk12) * np.dot(IIq4, CCq12)
    prodC13 = np.dot(IIk4, CCk13) * np.dot(IIq4, CCq13)
    prodC14 = np.dot(IIk4, CCk14) * np.dot(IIq4, CCq14)
    RHS = prodC1 * CCp1 + prodC2 * CCp2 + prodC3 * CCp3 + prodC4 * CCp4 + prodC5 * CCp5 + prodC6 * CCp6 + prodC7 * CCp7 + prodC8 * CCp8 + prodC9 * CCp9 + prodC10 * CCp10 + prodC11 * CCp11 + prodC12 * CCp12 + prodC13 * CCp13 + prodC14 * CCp14

    prodL1 = np.dot(IIk4, VVk1 * IIk4) * np.dot(IIq4, VVq1 * IIq4)
    prodL2 = np.dot(IIk4, VVk2 * IIk4) * np.dot(IIq4, VVq2 * IIq4)
    prodL3 = np.dot(IIk4, VVk3 * IIk4) * np.dot(IIq4, VVq3 * IIq4)
    prodL4 = np.dot(IIk4, VVk4 * IIk4) * np.dot(IIq4, VVq4 * IIq4)
    LHS = prodL1 * VVp1 + prodL2 * VVp2 + prodL3 * VVp3 + prodL4 * VVp4

    IIp4 = RHS / LHS

    # compute IIq4
    prodC1 = np.dot(IIk4, CCk1) * np.dot(IIp4, CCp1)
    prodC2 = np.dot(IIk4, CCk2) * np.dot(IIp4, CCp2)
    prodC3 = np.dot(IIk4, CCk3) * np.dot(IIp4, CCp3)
    prodC4 = np.dot(IIk4, CCk4) * np.dot(IIp4, CCp4)
    prodC5 = np.dot(IIk4, CCk5) * np.dot(IIp4, CCp5)
    prodC6 = np.dot(IIk4, CCk6) * np.dot(IIp4, CCp6)
    prodC7 = np.dot(IIk4, CCk7) * np.dot(IIp4, CCp7)
    prodC8 = np.dot(IIk4, CCk8) * np.dot(IIp4, CCp8)
    prodC9 = np.dot(IIk4, CCk9) * np.dot(IIp4, CCp9)
    prodC10 = np.dot(IIk4, CCk10) * np.dot(IIp4, CCp10)
    prodC11 = np.dot(IIk4, CCk11) * np.dot(IIp4, CCp11)
    prodC12 = np.dot(IIk4, CCk12) * np.dot(IIp4, CCp12)
    prodC13 = np.dot(IIk4, CCk13) * np.dot(IIp4, CCp13)
    prodC14 = np.dot(IIk4, CCk14) * np.dot(IIp4, CCp14)
    RHS = prodC1 * CCq1 + prodC2 * CCq2 + prodC3 * CCq3 + prodC4 * CCq4 + prodC5 * CCq5 + prodC6 * CCq6 + prodC7 * CCq7 + prodC8 * CCq8 + prodC9 * CCq9 + prodC10 * CCq10 + prodC11 * CCq11 + prodC12 * CCq12 + prodC13 * CCq13 + prodC14 * CCq14

    prodL1 = np.dot(IIk4, VVk1 * IIk4) * np.dot(IIp4, VVp1 * IIp4)
    prodL2 = np.dot(IIk4, VVk2 * IIk4) * np.dot(IIp4, VVp2 * IIp4)
    prodL3 = np.dot(IIk4, VVk3 * IIk4) * np.dot(IIp4, VVp3 * IIp4)
    prodL4 = np.dot(IIk4, VVk4 * IIk4) * np.dot(IIp4, VVp3 * IIp4)
    LHS = prodL1 * VVq1 + prodL2 * VVq2 + prodL3 * VVq3 + prodL4 * VVq4

    IIq4 = RHS / LHS

VVk1 = np.dot(Yinv, IIk1)
VVk2 = np.dot(Yinv, IIk2)
VVk3 = np.dot(Yinv, IIk3)
VVk4 = np.dot(Yinv, IIk4)

Vfinal1 = VVk1 * IIp1[50] * IIq1[200] + VVk2 * IIp2[50] * IIq2[200] + VVk3 * IIp3[50] * IIq3[200] + VVk4 * IIp4[50] * IIq4[200]
print(Vfinal1 + np.dot(Yinv, I0_pq))
    
# -----------------END OUTER LOOP NUMBER 2-----------------

VVk1 = np.conj(VVk1)
VVk2 = np.conj(VVk2)
VVk3 = np.conj(VVk3)
VVk4 = np.conj(VVk4)

VVp1 = np.conj(IIp1)
VVp2 = np.conj(IIp2)
VVp3 = np.conj(IIp3)
VVp4 = np.conj(IIp4)

VVq1 = np.conj(IIq1)
VVq2 = np.conj(IIq2)
VVq3 = np.conj(IIq3)
VVq4 = np.conj(IIq4)

# VVr = [VVk1, VVp1, VVq1]

SSk1 = np.conj(Kk1)
SSp1 = np.conj(Pp1)
SSq1 = np.conj(Qq1)
SSk2 = np.conj(Kk2)
SSp2 = np.conj(Pp2)
SSq2 = np.conj(Qq2)

# SSr = [[SSk1, SSp1, SSq1], [SSk2, SSp2, SSq2]]

CCk1 = SSk1
CCp1 = SSp1
CCq1 = SSq1
CCk2 = SSk2
CCp2 = SSp2
CCq2 = SSq2

# CCr = [[CCk1, CCp1, CCq1], [CCk2, CCp2, CCq2]
