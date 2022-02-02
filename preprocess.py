from re import template
import numpy as np
from numpy.ma.core import count
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from timeit import default_timer as timer

class PCA_Preprocessor:
    # A relatively simple PCA preprocessor
    # used to decompose the input into its most prominent features
    
    def __init__(self) -> None:
        pass

    def preprocess(self, input, n_pois):
        pca = PCA(n_components=n_pois)
        processed_input = pca.fit_transform(input)

        return processed_input, [0,0]

class LDA_Preprocessor:
    # An LDA preproccessor more sophisticated than PCA
    # Also used to decompose the input to its most prominent features
    # But uses the labels as well instead of only the data

    def __init__(self) -> None:        
        pass
    
    def preprocess(self, x_input, y_input, n_pois):
        lda = LinearDiscriminantAnalysis(n_components=n_pois)
        processed_input = lda.fit_transform(x_input, y_input)

        return processed_input, [0,0]

class SOST_Preprocessor:
    # Sum Of Squared T-Test preprocessor
    # Creates a T-test for each feature to select the most prominent ones
    def __init__(self, amount_of_values) -> None:
        self.HW = [bin(a).count("1") for a in range(256)]
        self.amount_of_values = amount_of_values
        pass

    def preprocess(self, x_input, y_input, n_poi, poi_spacing=20):
        if self.amount_of_values == 9:
            trace_groups = [[] for _ in range(self.amount_of_values)]      
            for i in range (len(x_input)):
                hw = self.HW[y_input[i]] 
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
        relevant_indices = np.full(n_poi, -1)
        check_idx = 1

        for i in range(n_poi):            
            if i > 0:           
                cur_poi = sorted[check_idx]
                while(not self.dist_to_pois(cur_poi, relevant_indices, poi_spacing)):                    
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
    
    def __init__(self) -> None:
        pass
    
    def preprocess(self, input):
        return input