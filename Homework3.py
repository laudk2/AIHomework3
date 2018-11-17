import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math


epsilon = 0.005
weights = [np.random.normal(), np.random.normal()]
weightsLog = [weights]

total_errors = []
testing_errors = []
input_count = 1


def display_plot_of_temperatures():
    trainingData1 = pd.read_csv("train_data_1.txt", header=None)
    trainingData2 = pd.read_csv("train_data_2.txt", header=None)
    trainingData3 = pd.read_csv("train_data_3.txt", header=None)
    testData = pd.read_csv("test_data_4.txt", header=None)

    ax = trainingData1.plot(x=0, y=1, linestyle=":", label="Day One", title="Plots of temperatures")
    trainingData2.plot(x=0, y=1, linestyle=":", ax=ax, label="Day Two")
    trainingData3.plot(x=0, y=1, linestyle=":", ax=ax, label="Day Three")
    testData.plot(x=0, y=1, linestyle=":", ax=ax, label="Day Four")
    ax.set_xlabel("Hour of the day (13 = 1PM)")
    ax.set_ylabel("Temperature (F)")






def calculate_error(data_frame, weights):
    sum = 0

    for index, row in data_frame.iterrows():
        calculated_output = row[0] * weights[0] + weights[1]
        sum += math.pow(calculated_output - row[1], 2)

    return math.sqrt(sum)

def calculate_weight_after_delta_d(current_weight, current_pattern, alpha=0.0005, k=0.5):
    net = current_pattern[0] * weights[0] + weights[1]

    output = net * 1
    delta_d = alpha * (current_pattern[1] - output)

    current_pattern[0] *= delta_d
    current_pattern[1] *= delta_d

    current_weight[0] += current_pattern[0]
    current_weight[1] += current_pattern[1]

    return current_weight

def learn(number_of_iterations):
    final_weights = weights

    trainingData1 = pd.read_csv("train_data_1.txt", header=None)
    trainingData2 = pd.read_csv("train_data_2.txt", header=None)
    trainingData3 = pd.read_csv("train_data_3.txt", header=None)
    testData = pd.read_csv("test_data_4.txt", header=None)

    train_df = trainingData1.append(trainingData2).append(trainingData3)

    for iter in range(0, number_of_iterations):
        print("iteration", iter)

        total_error = calculate_error(testData, final_weights)

        testing_errors.append(total_error)

        if epsilon > total_error:
            break

        # For each element in the data_frame `train_df`
        for index, row in train_df.iterrows():
            new_weights = calculate_weight_after_delta_d(weights, row)

            for i in range(0, input_count):
                weights[i] = new_weights[i]

        final_weights = weights

        weightsLog.append(final_weights)
    return final_weights

def main():
    display_plot_of_temperatures()
    learn(number_of_iterations=100)
    plt.plot(testing_errors)

if __name__ == "__main__":
        main()