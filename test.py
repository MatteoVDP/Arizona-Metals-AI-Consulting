#https://realpython.com/python-ai-neural-network/

import numpy as np
from neural_network import NeuralNetwork
import pandas as pd
import matplotlib.pyplot as plt

def normalize_data(data):
    # Normalize the data to the range [0, 1]
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    range_vals = max_vals - min_vals
    # Handle the case where the range is zero
    range_vals[range_vals == 0] = 1  # Set range to 1 where it's zero
    return (data - min_vals) / range_vals

def main():

    #read excel file
    df = pd.read_excel("AI_Brandon_Soils_Selection_Kay_Combined.xlsx")
    #pull mineralization values
    min = df.iloc[:, 1].tolist()
    mineralization= np.array(min)
    #remove unneeded columns
    columns_to_remove_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    columns_to_remove = [df.columns[i] for i in columns_to_remove_indices]
    # Drop specified columns from the DataFrame
    df = df.drop(columns=columns_to_remove)
    # Convert the DataFrame to a list of lists
    data = df.values.tolist()
    data1 = np.array(data)
    #set all negative values to zero
    data1[data1 < 0] = 0
    data_array = normalize_data(data1)

    #parameters
    learning_rate = 0.1
    runs = 10000

    #train 
    neural_network = NeuralNetwork(learning_rate)
    training_error = neural_network.train(data_array, mineralization, runs)
    plt.plot(training_error)
    plt.xlabel("Iterations")
    plt.ylabel("Error for all training instances")
    plt.savefig("cumulative_error.png")


    #make predictions
    input1 = [0.003, 0.21, 1.39, 12.7, -10.0, 130.0, 0.4, 0.17, 0.79, 0.23, 30.4, 14.8, 28.0, 1.49, 47.0, 3.02, 4.81, 0.08, 0.07, 0.04, 0.021, 0.39, 14.4, 17.0, 0.78, 593.0, 0.63, 0.02, 0.76, 25.3, 680.0, 17.2, 17.7, -0.001, 0.01, 0.89, 5.3, 0.2, 0.4, 47.2, -0.01, 0.05, 3.0, 0.061, 0.1, 0.51, 55.0, 0.24, 8.0, 82.0, 2.5]
    input2 = [0.006, 0.11, 2.31, 81.8, -10.0, 30.0, 0.26, 0.08, 11.4, 0.34, 9.39, 26.6, 57.0, 0.5, 64.4, 6.37, 8.87, 0.06, 0.05, 0.11, 0.036, 0.08, 4.4, 19.1, 1.4, 1090.0, 0.67, -0.01, 0.13, 51.0, 470.0, 5.4, 4.6, 0.001, 0.01, 6.6, 12.0, 0.4, -0.2, 106.5, -0.01, 0.05, 1.1, 0.007, 0.04, 0.36, 123.0, 0.07, 4.51, 100.0, 1.7]
    input1_normalized = normalize_data(np.array(input1).reshape(1, -1))
    input2_normalized = normalize_data(np.array(input2).reshape(1, -1))
    # Make predictions
    figure1 = neural_network.predict(input1_normalized)
    figure2 = neural_network.predict(input2_normalized)
    print ("Should Hit:")
    print (figure1)
    print ("Should Miss:")
    print(figure2)
    
if __name__ == "__main__":
    main()