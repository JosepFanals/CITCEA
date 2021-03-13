import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Zf = 0.03 + 0.00 * 1j  # fault impedance
Z1 = 0.01 + 0.10 * 1j  # Za in the drawings
Zth = 0.01 + 0.05 * 1j  # old Zth in the drawing

Vth_o = 1  # positive sequence voltage. Its angle will change, but not relevant

Zs_i = 6.674e-5 + 1j * 2.597e-4  # series impedances in pu/km
Zp_i = - 1j * 77.372  # parallel impedance in pu.km

dist = 1  # 1 km

Vth_vec = []
Z2_vec = []
dist_vec = []

for dist in range(1, 100):
    Zp = Zp_i / dist 
    Zs = Zs_i * dist

    Vth = Vth_o * Zp * Zp / (2 * Zth * Zp + Zp * Zs + Zp * Zp + Zth * Zs) 
    Z2 = (Zp * Zp * Zth + Zs * Zp * Zp + Zth * Zs * Zp) / (2 * Zp * Zth + Zp * Zp + Zs * Zp + Zth * Zs)

    Vth_vec.append(Vth)
    Z2_vec.append(Z2)
    dist_vec.append(dist)

plt.plot(dist_vec, np.abs(Z2_vec))
plt.show()

def make_csv(x_vec, y_vec, file_name):
    df = pd.DataFrame(data=[x_vec, y_vec]).T
    df.columns = ['x', 'y']
    df.to_csv(file_name, index=False)


make_csv(dist_vec, np.real(Vth_vec), 'Data/cable/Vth_re.csv')
make_csv(dist_vec, np.imag(Vth_vec), 'Data/cable/Vth_im.csv')
make_csv(dist_vec, np.abs(Vth_vec), 'Data/cable/Vth_abs.csv')

make_csv(dist_vec, np.real(Z2_vec), 'Data/cable/Zth_re.csv')
make_csv(dist_vec, np.imag(Z2_vec), 'Data/cable/Zth_im.csv')
make_csv(dist_vec, np.abs(Z2_vec), 'Data/cable/Zth_abs.csv')



# checking

# dist = 50
# Zp = Zp_i / dist 
# Zs = Zs_i * dist

# Vx = 1 / Zth / (1 / (Zs + Zp) + 1 / Zp + 1 / Zth)
# print(Vx)
# Voo = Vx * Zp / (Zs + Zp)

# # Voo equal to Vth


# Vth = Vth_o * Zp * Zp / (2 * Zth * Zp + Zp * Zs + Zp * Zp + Zth * Zs) 
# Z2 = (Zp * Zp * Zth + Zs * Zp * Zp + Zth * Zs * Zp) / (2 * Zp * Zth + Zp * Zp + Zs * Zp + Zth * Zs)

# print(Vth, Voo)

# Ioo2 = Vth / (Z2 + 10)
# print(Ioo2)

# Zoo2 = 10 * Zp / (10 + Zp)
# Vx = 1 / Zth / (1 / (Zs + Zoo2) + 1 / Zp + 1 / Zth)
# Voo2 = Vx * Zoo2 / (Zoo2 + Zs)
# print(Voo2 / 10)