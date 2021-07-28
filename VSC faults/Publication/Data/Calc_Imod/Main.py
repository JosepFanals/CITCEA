import numpy as np
import pandas as pd

folder = 'LL_cable_1conv/'
file1 = 'gc_LL_Ip1re.csv'
file2 = 'gc_LL_Ip1im.csv'
file_result = 'res_gc_LL_Ip.csv'

df1 = pd.read_csv(folder+file1)
df2 = pd.read_csv(folder+file2)
df3 = df1.copy()
df3['y2'] = df2['y'].to_numpy()
df3['mod'] = ""
for ll in range(len(df3)):
	df3.iloc[ll, 3] = np.sqrt(df3.iloc[ll, 1] ** 2 + df3.iloc[ll, 2] ** 2)
df4 = df3.drop(columns=['y','y2'])
df5 = df4.rename(columns={"mod": "y"})

print(df5)
# print(folder_result)
df5.to_csv(folder+file_result)
