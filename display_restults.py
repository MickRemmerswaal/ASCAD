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

total_results = []
averaged_results = []

for i in range(3,8):
    data_loc = "results_fixed_key_byte"+ str(i) +".csv"
    results = load_data(data_loc)
    total_results.append(results)

# Calculate average scores
for i in range(5):
    sost_average = calc_average(total_results, range(6))
    LDA_average = calc_average(total_results, range(6, 21, 2))
    PCA_average = calc_average(total_results, range(7, 22, 2))
sost_labels = ["2", "4", "10", "20", "50", "100"]
for i in range(0, len(sost_average)):
    plt.plot(sost_average[i], label=(sost_labels[i] +" POIs"))

plt.title("Averege Guessing Entropy of LDA Point of Interest Selection with Variable N POIs")
plt.ylim([0, 256])
plt.xlim([0, 1000])
plt.legend()
plt.show(block=True)



