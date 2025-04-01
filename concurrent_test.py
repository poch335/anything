from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
from time import time
import concurrent.futures

data_path = 'C:\\code\\Data\\Raptor_rawdata\\procrustes'
folder_path = 'C:\\code\\Data\\230617\\procrustes\\EPD_result'


def translate_mean_to_center(np_set):
    np_set -= np.mean(np_set, 0)
    return np_set


def scaling(X: np.array):
    temp_X = translate_mean_to_center(X)
    temp_X = temp_X / np.linalg.norm(temp_X)
    return temp_X


def orthogonal_transform(args):
    x, y = args
    U, S, Vh = np.linalg.svd((y.T @ x).T)
    R = U @ Vh
    scale = np.sum(S)
    return R, scale


def main():
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    
    os.chdir(data_path)
    
    file_list = glob.glob('*.csv')
    
    log_disparity = {}
    log_rad = {}
    
    for f in file_list:
        temp_file = pd.read_csv(f, delimiter=',')
        log_disparity[f] = []
        log_rad[f] = []
    
        n = 5
        i = 0
    
        for x in tqdm(range(n, temp_file.shape[0])):
            data_new = np.array(temp_file.iloc[x-n:x, :], dtype=np.float64)
    
            if i == 0:
                data_temp = scaling(data_new).copy()
                i += 1
                continue
    
            data_new = scaling(data_new)
    
            with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
                results = executor.map(orthogonal_transform, [(data_temp, data_new)])
                R, s = next(results)
    
            disparity = np.linalg.norm(data_temp - data_new)
            log_disparity[f].append(disparity)
            log_rad[f].append(np.arccos(R[0, 0]))
    
            data_temp = data_new.copy()
    
        plt.title(f'EPD_{os.path.basename(f)}')
        plt.plot(np.arange(len(log_disparity[f])), log_disparity[f])
        plt.show()
        plt.title(f'EPD_{os.path.basename(f)}')
        plt.plot(np.arange(len(log_rad[f])), log_rad[f])
        plt.show()
            
if __name__ == '__main__':
    main()