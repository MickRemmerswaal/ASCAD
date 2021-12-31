import h5py as h5
import matplotlib.pyplot as plt

file = h5.File('ATMEGA_AES_v1\ATM_AES_v1_fixed_key\ASCAD_data\ASCAD_databases\ASCAD.h5', 'r')

print(list(file.keys()))

group_attack = file['Attack_traces']
group_profiling = file['Profiling_traces']

#dataset_attack = group_attack
#dataset_profiling = group_profiling['Profiling_traces']

data_items = list(group_attack.items())
for item in data_items:
    print(item[0])

labels = list(group_attack.get(data_items[0][0]))
metadata = list(group_attack.get(data_items[1][0]))
traces = list(group_attack.get(data_items[2][0]))

plt.plot(traces[0])
plt.show()