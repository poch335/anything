import pywt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def DWT(signal, thresh = 0.63, wavelet="db5"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

# data_path = 'C:\\code\\Data\\modified_230603\\20wb\\test'

data_path = 'C:\\code\\DWT'
folder_path = 'C:\\code\\DWT\\231226_DWT'

if not os.path.exists(folder_path):
    os.mkdir(folder_path)


os.chdir(data_path)
plot_file = glob.glob('*.csv')

for f in plot_file:
    temp_plot = pd.read_csv(f, delimiter = ',')
    #cnt = np.arange(temp_plot.shape[0])
    
    
        

    # signal = temp_plot.iloc[:, 1:].values
    data = np.zeros((temp_plot.shape[0], temp_plot.shape[1]))     
    
    for i in range(temp_plot.shape[1]):
        filtering = DWT(temp_plot.iloc[:, i], 0.4)
        data[:, i] = filtering[:-1]
        
     
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(signal, color="b", alpha=0.5, label='original signal')
    rec = DWT(signal, 0.4)
    ax.plot(rec, 'k', label='DWT smoothing}', linewidth=2)
    ax.legend()
    ax.set_title('Removing High Frequency Noise with DWT', fontsize=18)
    ax.set_ylabel('Signal Amplitude', fontsize=16)
    ax.set_xlabel('Sample No', fontsize=16)
    plt.show()
    
    integrate = np.stack((signal, rec[:-1]), axis = 1)

    np.savetxt(f'{folder_path}/DWT_result_{os.path.basename(f)}', filtering, delimiter = ',', fmt = '%f')    
    