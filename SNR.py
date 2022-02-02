import os.path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

# def create_z_score_norm(dataset):
#     z_score_mean = np.mean(dataset, axis=0)
#     z_score_std = np.std(dataset, axis=0)
#     return z_score_mean, z_score_std


# def apply_z_score_norm(dataset, z_score_mean, z_score_std):
#     for index in range(len(dataset)):
#         dataset[index] = (dataset[index] - z_score_mean) / z_score_std

def CalculateSNR(l, IntermediateData):
    trace_length = l.shape[1]
    mean = np.zeros([256, trace_length])
    var = np.zeros([256, trace_length])
    cpt = np.zeros(256)
    i = 0

    for trace in l:
        # classify the traces based on its SBox output
        # then add the classified traces together
        mean[IntermediateData[i]] += trace
        var[IntermediateData[i]] += np.square(trace)
        # count the trace number for each SBox output
        cpt[IntermediateData[i]] += 1
        i += 1
    for i in range(256):
        # average the traces for each SBox output
        mean[i] = mean[i] / cpt[i]
        # variance  = mean[x^2] - (mean[x])^2
        var[i] = var[i] / cpt[i] - np.square(mean[i])
    # Compute mean [var_cond] and var[mean_cond] for the conditional variances and means previously processed
    # calculate the trace variation in each points
    varMean = np.var(mean, 0)
    # evaluate if current point is stable for all S[p^k] cases
    MeanVar = np.mean(var, 0)
    return (varMean, MeanVar)

  
def SNR_cal(trace_directory, x_range, save_root):
    # here, enter your own datapath
    data_dir = trace_dir = trace_directory
    n_samples = 50000
    targeted_sbox_index = 2
    targeted_keyExpansion_index = 42

    print("Read Traces, plaintexts, keys, masks values and initialize the labels used afterwards")
    f = h5py.File(trace_dir, "r")
    trace = h5py.File(trace_dir, "r")
    l = np.array(trace['Profiling_traces/traces'][:n_samples, :], dtype=np.float32)

    data = np.array(f['Profiling_traces/metadata'][:n_samples])

    k = np.array(data['key'][:, targeted_sbox_index], dtype=np.int16)
    p = np.array(data['plaintext'][:, targeted_sbox_index], dtype=np.int16)
    r = np.array(data['masks'][:, targeted_sbox_index], dtype=np.int16)
    print(data['masks'][0])
    rout = np.array(data['masks'][:, 15], dtype=np.int16)

    print("Calculate intermediate data")
    HW = np.array([bin(n).count("1") for n in range(0, 256)])
    SboxOut_withMaskRout = Sbox[k ^ p] ^ rout
    hw_SboxOut_withMaskRout = HW[SboxOut_withMaskRout]
    SboxOut_withoutMaskRout = SboxOut_withMaskRout ^ rout
    hw_SboxOut_withoutMaskRout = HW[SboxOut_withoutMaskRout]
    SboxOut_withMaskR = Sbox[k ^ p] ^ r

    print("Plot the original traces")
    plt.set_cmap('Blues')
    plt.tight_layout()
    # plt.plot(l[0])
    # plt.show()

    print("Calculate SNR and plot the data")
    IntermediateData = [SboxOut_withMaskRout, SboxOut_withoutMaskRout, SboxOut_withMaskR, rout, r]
    FigureLable = ['SboxOut_withMaskRout', 'SboxOut_withoutMaskRout', 'SboxOut_withMaskR', 'rout', 'r']
    IntermediateData = [SboxOut_withMaskR, r]
    FigureLable = [r'$Sbox(p_2 \oplus k_2) \oplus r_2$', r'$r_2$']
    FigureArray = []
    for idx in range(len(IntermediateData)):
        varMean, MeanVar = CalculateSNR(l, IntermediateData[idx])
        snr_plot, = plt.plot(x_range, (varMean / MeanVar), label=FigureLable[idx])

        # labels_for_snr = [label for label in zip(np.asarray(IntermediateData[idx]))]
        # snr = SNR(np=1, ns=len(l[0]), nc=256)
        # snr.fit_u(np.array(l, dtype=np.int16), x=np.array(labels_for_snr, dtype=np.uint16))
        # snr_val = snr.get_snr()
        # snr_plot, = plt.plot(x_range, snr_val[0], label=FigureLable[idx])
        
        FigureArray.append(snr_plot)
    plt.grid(ls='--')
    plt.xlabel('Time Samples')
    plt.ylabel('SNR')
    plt.locator_params(axis='x', nbins=6)
    plt.xlim(min(x_range), max(x_range))
    plt.ylim(0)
    plt.legend(handles=FigureArray, loc=2)
    plt.savefig(save_root)
    plt.show()
    plt.clf()

if __name__ == "__main__":
    # Linux
    # Base_varK_desync0
    matplotlib.rcParams.update({'font.size': 12})
    ascad_database = "/usr/local/home/wu/ownCloud/PhD/Data/ASCAD_rand/ascad-variable.h5"
    save_root = '/usr/local/home/wu/ownCloud/PhD/Research/Similiarity learning/04 - Result/'
    #new_traces_file = "/usr/local/home/wu/ownCloud/PhD/Data/ASCAD/Trace/Base/4000_varK_all.h5"

    # Windows
    #ascad_database = "F:/surfdrive/PhD/Data/ASCAD/Traces/ascad-variable.h5"
    #ascad_database = "F:/surfdrive/PhD/Data/ASCAD/Noisy traces/Noisy_shuffling.h5"
    #ascad_database = "F:/surfdrive/PhD/Data/ASCAD/Denoised traces/Denoised_all_CAE - Copy.h5"
    #ascad_database = "F:/surfdrive/PhD/Data/ASCAD/Base_desync0.h5"
    #ascad_database = "F:/surfdrive/PhD/Data/ASCAD_rand/ascad-variable.h5"

    SNR_cal(ascad_database, range(44000, 45400), save_root)
