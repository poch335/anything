import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler

# Function to plot the U-Matrix
def plot_umatrix(som, ax):
    weights = som.get_weights()
    x, y, _ = weights.shape
    u_matrix = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            neighbors = []
            if i > 0:
                neighbors.append(weights[i-1, j])  # Up
            if i < x-1:
                neighbors.append(weights[i+1, j])  # Down
            if j > 0:
                neighbors.append(weights[i, j-1])  # Left
            if j < y-1:
                neighbors.append(weights[i, j+1])  # Right

            distances = [np.linalg.norm(weights[i, j] - neighbor) for neighbor in neighbors]
            u_matrix[i, j] = np.mean(distances)
    
    ax.imshow(u_matrix, cmap='bone')
    ax.set_title('U-Matrix')
    ax.set_xticks(np.arange(0, x, 1))
    ax.set_yticks(np.arange(0, y, 1))
    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5)

# Load the uploaded file
file_path = 'test'
files = glob.glob(f'{file_path}/*.csv')

for file in files:
    temp_data = pd.read_csv(file)
    
    
    data = temp_data.iloc[:, 1100:1650]
    # Extract the wavelengths from the column names
    wavelengths = data.columns.values
     
    # Normalize each row of the entire data using MinMaxScaler
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.values)
    
    # Transpose the normalized data
    data_transposed = data_normalized.transpose()
    
    # Check the shapes of the data
    print(f'File: {file}')
    print(f'Original data shape: {data.shape}')
    print(f'Normalized data shape: {data_normalized.shape}')
    print(f'Transposed normalized data shape: {data_transposed.shape}')
    
    # Initialize and train the SOM
    np.random.seed(42)
    som = MiniSom(x=30, y=30, input_len=data_transposed.shape[1], sigma=1.0, learning_rate=1, random_seed=42)
    som.random_weights_init(data_transposed)
    som.train_random(data_transposed, num_iteration=20000)
    
    # Plotting the results
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_umatrix(som, ax)
    
    # Plot the data points on the U-Matrix
    for cnt, xx in enumerate(data_transposed):
        w = som.winner(xx)
        ax.plot(w[0] + 0.5, w[1] + 0.5, 'o', markerfacecolor=plt.cm.rainbow(cnt / len(data_transposed)), 
                markeredgecolor='none', markersize=5, alpha=0.7)
    
    # Add legend for the wavelengths
    unique_wavelengths = np.unique(wavelengths)
    colors = [plt.cm.rainbow(i / len(unique_wavelengths)) for i in range(len(unique_wavelengths))]
    for wavelength, color in zip(unique_wavelengths, colors):
        ax.plot([], [], 'o', color=color, label=str(wavelength))
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', title='Wavelengths')
    
    plt.title(f'SOM for Data - {file}')
    plt.xlabel('SOM x-dimension')
    plt.ylabel('SOM y-dimension')
    plt.show()
