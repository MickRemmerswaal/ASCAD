import numpy as np
from numpy.ma.core import count
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import tensorflow as tf
from autoencoder import _Conv1DTranspose, ConvLayer, DeconvLayer

class PCA_Preprocessor:
    # A relatively simple PCA preprocessor
    # used to decompose the input into its most prominent features
    
    def __init__(self, n_pois) -> None:
        self.pca = PCA(n_components=n_pois)
        pass

    def preprocess(self, temp_input, temp_labels, atk_input):
        processed_input_temp = self.pca.fit_transform(temp_input)
        processed_input_atk = self.pca.transform(atk_input)

        return processed_input_temp, processed_input_atk

class LDA_Preprocessor:
    # An LDA preproccessor more sophisticated than PCA
    # Also used to decompose the input to its most prominent features
    # But uses the labels as well instead of only the data

    def __init__(self, n_pois) -> None:
        self.lda = LinearDiscriminantAnalysis(n_components=n_pois)
        pass
    
    def preprocess(self, temp_input, temp_labels, atk_input):
        processed_input_temp = self.lda.fit_transform(temp_input, temp_labels)
        processed_input_atk = self.lda.transform(atk_input)

        return processed_input_temp, processed_input_atk

class SOST_Preprocessor:
    # Sum Of Squared T-Test preprocessor
    # Creates a T-test for each feature to select the most prominent ones
    def __init__(self, amount_of_values, n_pois, poi_spacing) -> None:
        self.HW = [bin(a).count("1") for a in range(256)]
        self.amount_of_values = amount_of_values
        self.n_pois = n_pois
        self.poi_spacing = poi_spacing
        pass

    def preprocess(self, x_input, y_input):
        if self.amount_of_values == 9:
            trace_groups = [[] for _ in range(self.amount_of_values)]      
            for i in range (len(x_input)):
                hw = y_input[i]
                trace_groups[hw].append(x_input[i])
        
        elif self.amount_of_values == 256:
            trace_groups = [[] for _ in range(self.amount_of_values)]      
            for i in range (len(x_input)):
                trace_groups[y_input[i]].append(x_input[i])
        else:
            print("Error: please choose either 9 or 256 as the amount of values")

        means = [[] for _ in range(self.amount_of_values)]
        var = [[] for _ in range(self.amount_of_values)]
        counts = [[] for _ in range(self.amount_of_values)]
        coefficients = np.zeros(len(x_input[0]))

        # Calculate means and variance of each label
        for i in range(self.amount_of_values):
            means[i] = np.mean(trace_groups[i], axis=0)
            var[i] = np.var(trace_groups[i], axis=0)
            counts[i] = len(trace_groups[i])

        # Calculate coefficients, with sum of squared differences
        for i in range(self.amount_of_values):
            for j in range(i):
                temp = pow(means[i] - means[j], 2)
                temp /= (var[i]/counts[i] + var[j]/counts[j])
                coefficients += temp

        # Select top-n coefficients to represent the POI's,
        # Adhereing to POI spacing to increase efficiency
        sorted =np.argsort(-coefficients)
        relevant_indices = np.full(self.n_pois, -1)
        check_idx = 1

        for i in range(self.n_pois):            
            if i > 0:           
                cur_poi = sorted[check_idx]
                while(not self.dist_to_pois(cur_poi, relevant_indices, self.poi_spacing)):                    
                    check_idx+=1
                    cur_poi = sorted[check_idx]

                relevant_indices[i] = cur_poi
                
            else:
                relevant_indices[0] = sorted[0]
        
        return x_input[:, relevant_indices], relevant_indices
    
    def dist_to_pois(self, cur_poi, all_pois, spacing):
        for poi in all_pois:
            if poi >= 0 and abs(cur_poi - poi) < spacing:
                return False
                
        return True

class DL_Preprocessor:
    
    def __init__(self, model_path) -> None:
        self.Encoder = tf.keras.models.load_model(model_path, custom_objects={'_Conv1DTranspose': _Conv1DTranspose,
                                                                              'ConvLayer'       : ConvLayer,
                                                                              'DeconvLayer'     : DeconvLayer})

    def preprocess(self, temp_input, atk_input):
        temp_out = self.Encoder.predict(temp_input)
        atk_out  = self.Encoder.predict(atk_input)
        return temp_out, atk_out
