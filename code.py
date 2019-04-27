import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import matplotlib

re_col_name = re.compile("(:?\')(.+)(\.)")

def read_itx(path):
    df = pd.DataFrame()
    df_list = []
    col_name_list = []
    col_name = ''
    re_col_name = re.compile("(:?\')(.+)(\.)")
    
    with open(path, 'r') as file:
        while True:
            df_length = 0
            df_length_prev = 0
            buffer = file.readline()
            if not buffer: 
                break
            if buffer.split('/')[0] == 'WAVES':
                df_list = []
                col_name = re_col_name.search(buffer).group(2).split('.')[0]
                buffer = file.readline()
                buffer = file.readline()
                while buffer != 'END':
                    df_list.append(float(buffer))
                    buffer = file.readline().strip()

                df[col_name] = df_list
    return df

data_list1 = []
foldername = 'folder_path'

for ind in range(41, 51):
    basename = 'Vbg_T_H_Sweep.'
    filename = basename + '{:06d}'.format(ind) + '.itx' 
    path = foldername + '/' + filename
    data_list1.append(read_itx(path))

VL = np.zeros((len(data_list1[0]), 10))
VH = np.zeros((len(data_list1[0]), 10))
I = np.zeros((len(data_list1[0]), 10))
B = np.zeros(10)
e = 1.6e-19
for i in range(10):
    VL[:, i] = data_list1[i]['V3X'].values
    VH[:, i] = data_list1[i]['V2X'].values
    I[:, i] = data_list1[i]['CurrentX'].values
    B[i] = data_list1[i]['Field'].values[0]/10000
I = I.mean(axis=1)
Vbg = data_list1[0]['Vbg'].values

CarrierDensity = np.zeros(len(data_list1[0]))
error = np.zeros(len(data_list1[0]))
for i in range(len(data_list1[0])):
    p, cov = np.polyfit(B, VH[i, :], 1, full=False, cov=True)
    CarrierDensity[i] = I[i] / (e * 1e16) * (1/p[0])
    # error of carrier density is calculated using the propagation of error, 
    # and assume 95% of confidence (+-1.96 sigma)
    error[i] = I[i] / (e * 1e16) * (np.sqrt(cov[0, 0])) * (1/p[0])**2 * 1.96 * 2

plt.plot(Vbg, abs(CarrierDensity))
plt.show()