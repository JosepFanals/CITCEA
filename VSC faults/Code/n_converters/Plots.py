import numpy as np
import pandas as pd

def fPlots(x, y, name):

    def make_csv(x_vec, y_vec, file_name):
        df = pd.DataFrame(data=[x_vec, y_vec]).T
        df.columns = ['x', 'y']
        df.to_csv(file_name, index=False)

    for ii in range(len(y)):
        make_csv(x, y[ii], name + str(ii) + '.csv')
