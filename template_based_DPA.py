import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import h5py as h5
import preprocess as prep

# Simple S-Box lookup table
sbox=(
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16) 

HW = [bin(a).count("1") for a in range(256)]

# Calculate intermediate value by lookup input XOR key for a certain byte index
def intermediate_val(input, key, byte_idx=0):
    return sbox[input[byte_idx] ^ key[byte_idx]]

def intermediate_val_att(input, key, byte_idx=0):
    return sbox[input[byte_idx] ^ key]

# Calculate Hamming weight of sbox output
def get_hamming_weight(input, key, byte_idx=0):
    return HW[intermediate_val(input, key, byte_idx)]

def get_hamming_weight_att(input, key, byte_idx=0):
    return HW[intermediate_val_att(input, key, byte_idx)]

# POI selection, reducing the dimensionality => increases efficiency and speed
def POI_selection(type, x_input, y_input, amount_of_values, n_pois=4):
    if type == "SOST": 
        proc = prep.SOST_Preprocessor(amount_of_values)
        snr = calc_snr(x_input, y_input, amount_of_values)
        return proc.preprocess(x_input, y_input, n_pois), snr
        
    elif type == "LDA":
        proc = prep.LDA_Preprocessor()
        snr = calc_snr(x_input, y_input, amount_of_values)
        return proc.preprocess(x_input, y_input, n_pois), snr

    elif type == "PCA":
        proc = prep.PCA_Preprocessor()
        snr = calc_snr(x_input, y_input, amount_of_values)
        return proc.preprocess(x_input, n_pois), snr
    else:
        print("Error: please select valid processor(SOST, LDA, PCA)")

# Calculate SNR for each group
def calc_snr(x_input, y_input, amount_of_values):
    # Group input based on labels
    grouped_input = [[] for _ in range(amount_of_values)]
    for i in range(len(x_input)):
        grouped_input[y_input[i]].append(x_input[i])

    # Calculate Means and Variance of each group
    snr_means = [np.array([]) for _ in range(amount_of_values)]
    snr_var = [np.array([]) for _ in range(amount_of_values)]

    for group in range(amount_of_values):
        snr_means[group] = np.mean(grouped_input[group], axis=0)
        snr_var[group] = np.var(grouped_input[group], axis=0)

    return np.divide(np.var(snr_means, axis=0), np.mean(snr_var, axis=0))

def create_templates(temp_input, temp_labels, amount_of_values):
    input_len = len(temp_input)
    n_pois = temp_input.shape[1]
    # Group input based on labels
    grouped_input = [[] for _ in range(amount_of_values)]
    for i in range(input_len):
        grouped_input[temp_labels[i]].append(temp_input[i])
    
    # Create mean vectors and covariance matrices for each group
    template_means = [np.array([]) for _ in range(amount_of_values)]    
    for group in range(len(grouped_input)):
        template_means[group] = np.mean(grouped_input[group], axis=0)

    template_covs = np.zeros((amount_of_values, n_pois, n_pois))

    for group in range(amount_of_values):
        for i in range(n_pois):
            for j in range(n_pois):            
                x = np.array(grouped_input[group])[:, i]
                y = np.array(grouped_input[group])[:, j]
                cov = np.cov(x,y)
                template_covs[group, i, j] = cov[0][1]


    return template_means, template_covs

def perform_attack(atk_input, atk_ptext, indices, template_means, template_cov_matrix, amount_of_values, attack_byte=1):    
    guess_proba = np.zeros(256)

    if amount_of_values == 9:
        for i in range(len(atk_input)):
            if not indices == [0,0]:
                cur_trace = atk_input[i, indices]
            else:
                cur_trace = atk_input[i]

            for key in range(256):
                hw = get_hamming_weight_att(atk_ptext[i], key, attack_byte)
                rv = multivariate_normal(template_means[hw], template_cov_matrix[hw], allow_singular=True)

                proba_key_guess = rv.pdf(cur_trace)
                guess_proba[key] += np.log(proba_key_guess)

            print(np.argsort(guess_proba)[-5:])

    elif amount_of_values == 256:
        for i in range(len(atk_input)):
            if not indices == [0,0]:
                cur_trace = atk_input[i, indices]
            else:
                cur_trace = atk_input[i]
                
            for key in range(256):
                val = intermediate_val_att(atk_ptext[i], key, attack_byte)
                rv = multivariate_normal(template_means[val], template_cov_matrix[val], allow_singular=True)

                proba_key_guess = rv.pdf(cur_trace)
                guess_proba[key] += np.log(proba_key_guess)

            print(np.argsort(guess_proba)[-5:])

if __name__ == "__main__":
    ####################################
    ### FIXED KEY TEMPLATE BASED DPA ###
    ####################################

    # Load in data
    file = h5.File('ATMEGA_AES_v1\ATM_AES_v1_fixed_key\ASCAD_data\ASCAD_databases\ATMega8515_raw_traces.h5', 'r')

    all_traces = file['traces']
    metadata = file['metadata']

    ## Metadata is build of four components
    ##  0 : Plain_text
    ##  1 : Cipher_text
    ##  2 : Key
    ##  3 : Mask

    all_ptext = [item[0] for item in metadata]
    all_ctext =  [item[1] for item in metadata]
    keys =  [item[2] for item in metadata]
    masks =  [item[3] for item in metadata]
    
    temp_traces = all_traces[0:8000]
    temp_ptext = all_ptext[0:8000]

    all_label_int = [intermediate_val(a, keys[0], 1) for a in all_ptext]
    temp_label_int = all_label_int[0:8000]
    temp_label_hw = [HW[a] for a in temp_label_int]

    atk_traces = all_traces[8000:8050]
    atk_ptext = all_ptext[8000:8050]

    # Process input for template creation    
    processed_input, indices, snr = POI_selection("PCA", temp_traces, temp_label_hw, 9)


    plt.plot(snr)
    plt.show(block=True)

    # Template creation
    template_means, template_covs = create_templates(processed_input, temp_label_hw, 9)

    ##################
    # Perform attack #
    ##################
    
    perform_attack(atk_traces, atk_ptext, indices, template_means, template_covs, 9, 2)
