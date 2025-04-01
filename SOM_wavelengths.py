import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler

# Load the uploaded file
file_path = 'test'
files = glob.glob(f'{file_path}/*.csv')

for file in files:
    data = pd.read_csv(file)
    
    # Extract the wavelengths from the column names
    wavelengths = data.columns.values
    
    # Normalize each row of the entire data using MinMaxScaler
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.values)  # Ensure to use .values to get a numpy array
    
    # Transpose the normalized data
    data_transposed = data_normalized.transpose()
    
    # Check the shapes of the data
    print(f'File: {file}')
    print(f'Original data shape: {data.shape}')
    print(f'Normalized data shape: {data_normalized.shape}')
    print(f'Transposed normalized data shape: {data_transposed.shape}')
    
    # Initialize and train the SOM
    np.random.seed(42)
    som = MiniSom(x=20, y=20, input_len=data_transposed.shape[1], sigma=1.0, learning_rate=0.25, random_seed=42)
    som.random_weights_init(data_transposed)
    som.train_random(data_transposed, num_iteration=10000)
    
    # Plotting the results
    plt.figure(figsize=(12, 12))
    for cnt, xx in enumerate(data_transposed):
        w = som.winner(xx)  # getting the winner
        plt.text(w[0] + 0.5, w[1] + 0.5, str(wavelengths[cnt]), color=plt.cm.rainbow(cnt / len(data_transposed)), 
                 fontdict={'weight': 'bold', 'size': 9})
    
    plt.xlim([0, 20])
    plt.ylim([0, 20])
    plt.title(f'SOM for Data - {file}')
    plt.xlabel('SOM x-dimension')
    plt.ylabel('SOM y-dimension')
    plt.grid()
    plt.show()
