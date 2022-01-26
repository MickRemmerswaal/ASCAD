import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

def snr(input_vector):

    output_vector = np.empty()
    
    return output_vector 


file = h5.File('ASCAD\ATMEGA_AES_v1\ATM_AES_v1_fixed_key\ASCAD_data\ASCAD_databases\ATMega8515_raw_traces.h5', 'r')

print(list(file.keys()))

traces = file['traces']
metadata = file['metadata']

## Metadata is build of four components
##  0 : Plain_text
##  1 : Cipher_text
##  2 : Key
##  3 : Mask

plain_texts = [item[0] for item in metadata]
cipher_texts =  [item[1] for item in metadata]
keys =  [item[2] for item in metadata]
masks =  [item[3] for item in metadata]
step = 100

plot1 = traces[0][47000:48000]
poi1 = []
plot2 = traces[1][47000:48000]
poi2 = []
plot3 = traces[2][47000:48000]
poi3 = []
plot4 = traces[3][47000:48000]
poi4 = []

for i in range(0, len(plot1), step):
    interval = plot1[i:i+step]
    poi1.append(np.argmax(interval) + i)

for i in range(0, len(plot2), step):
    interval = plot2[i:i+step]
    poi2.append(np.argmax(interval) + i)

for i in range(0, len(plot3), step):
    interval = plot3[i:i+step]
    poi3.append(np.argmax(interval) + i)

for i in range(0, len(plot4), step):
    interval = plot4[i:i+step]
    poi4.append(np.argmax(interval) + i)


fig, axs = plt.subplots(4, sharey=True)
#axs[0].plot(plot1, '-bo', markevery=poi1, markersize=6, markerfacecolor='k')
axs[0].plot(plot1, 'b')
#axs[0].axvspan(125, 148, color='red', alpha=0.5)
#axs[0].axvspan(743, 832, color='green', alpha=0.5)
axs[0].axvspan(125, 168, color='blue', alpha=0.5)

#axs[1].plot(plot2, '-go', markevery=poi2, markersize=6, markerfacecolor='k')
axs[1].plot(plot2, 'g')
#axs[1].axvspan(125, 148, color='red', alpha=0.5)
#axs[1].axvspan(743, 832, color='green', alpha=0.5)
axs[1].axvspan(705, 748, color='blue', alpha=0.5)

#axs[2].plot(plot3, '-ro', markevery=poi3, markersize=6, markerfacecolor='k')
axs[2].plot(plot3, 'r')
#axs[2].axvspan(125, 148, color='red', alpha=0.5)
#axs[2].axvspan(743, 832, color='green', alpha=0.5)
axs[2].axvspan(365, 408, color='blue', alpha=0.5)


#axs[3].plot(plot4, '-co', markevery=poi4, markersize=6, markerfacecolor='k')
axs[3].plot(plot4, 'c')
#axs[3].axvspan(125, 148, color='red', alpha=0.5)
#axs[3].axvspan(743, 832, color='green', alpha=0.5)
axs[3].axvspan(225, 268, color='blue', alpha=0.5)

for ax in axs.flat:
    ax.set(xlabel='Time', ylabel='PC')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.show(block=True)
