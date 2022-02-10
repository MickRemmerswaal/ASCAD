import matplotlib.pyplot as plt
import numpy as np


def load_data(data_loc):
    return np.genfromtxt(data_loc)

def average_all(files):
    return


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



# 0-5 == SOST   
# 6 8 10 12 14 16 18 20 LDA
# 7 9 11 13 15 17 19 21 PCA


plt.show(block=True)
