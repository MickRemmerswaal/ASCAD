import matplotlib.pyplot as plt
import numpy as np


def load_data(data_loc):
    return np.genfromtxt(data_loc, delimiter=',')

def calc_average(files, range):
    averages = []
    for i in range:
        to_average_rows = np.asarray(files)[:, i]
        average = np.round(np.average(to_average_rows, axis=0))
        averages.append(average)
    return averages

def display_row_data(data_file, row_nr):
    data_row = data_file.iloc[row_nr].values
    
    _, axs = plt.subplots(1, sharey=False)
    axs.plot(data_row)

def calc_succes_rate(files, range, index):
    # files: total results of experiments
    # range: range of indices which apply to current technique, e.g. range(6) for SOST
    #        first 5 experiments are bound to SOST
    # index: denotes amount of attack traces need
    #        e.g. index = 100 => we check if on index 100 there exists a 0.0 and denote that as a succes
    rates = []
    for i in range:
        selection_rows = np.asarray(files)[:, i]
        count = np.count_nonzero(selection_rows[:, (index-1)] == 0.0)
        rate = np.divide(count, 5)
        rates.append(rate)
    return rates

total_results = []
averaged_results = []

for i in range(3,8):
    data_loc = "results_fixed_key_byte"+ str(i) +".csv"
    results = load_data(data_loc)
    np.asarray(total_results.append(results))

# Calculate average scores
sost_average = calc_average(total_results, range(6))
LDA_average = calc_average(total_results, range(6, 21, 2))
PCA_average = calc_average(total_results, range(7, 22, 2))



# Calculating Succes rate

succes_rate_indices = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

SOST_succes_rate = []
PCA_succes_rate = []
LDA_succes_rate = []
AE_succes_rate = []

# Check for each index and each technique
# what % of the runs were able to have their guessing entropy at 0
for index in succes_rate_indices:
    # SOST
    SOST_succes_rate.append(calc_succes_rate(total_results, range(6), index))
    # PCA
    PCA_succes_rate.append(calc_succes_rate(total_results, range(7, 22, 2), index))
    # LDA
    LDA_succes_rate.append(calc_succes_rate(total_results,  range(6, 21, 2), index  ))


total_results = []
averaged_results = []

for i in range(3,8):
    data_loc = "results_AE_fixed_key_byte"+ str(i) +".csv"
    results = load_data(data_loc)
    total_results.append(results)

AE_average = calc_average(total_results, range(6))

sost_labels = ["2", "4", "10", "20", "50", "100"]

for i in succes_rate_indices:
    # AE Success rate
    AE_succes_rate.append(calc_succes_rate(total_results, range(6), index))


'''
# Plot results
for i in range(0, len(LDA_average)):
    plt.plot(LDA_average[i], label=(str(i+1) +" POIs"), linewidth=2)

plt.ylim([0, 256])
plt.xlim([0, 1000])
plt.ylabel("Guessing entropy")
plt.xlabel("Number of attack traces")
plt.legend()
plt.show(block=True)


'''

