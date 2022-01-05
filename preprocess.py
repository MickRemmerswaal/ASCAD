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
    
    def preprocess(self, variance, input):
        pca = PCA(variance)
        
        processed_input = pca.fit_transform(input)
        return processed_input

class LDA_Preprocessor:
    # An LDA preproccessor more sophisticated than PCA
    # Also used to decompose the input to it's most prominent features
    # But uses the labels as well instead of only the data


    def __init__(self) -> None:        
        pass

    def preprocess(self, x_input, y_input):
        lda = LinearDiscriminantAnalysis()
        
        processed_input = lda.fit_transform(x_input, y_input)
        return processed_input

class SOST_Preprocessor:
    # Sum Of Squared T-Test preprocessor
    # Creates a T-test for each feature to select the most prominent ones
    def __init__(self, amount_of_values) -> None:
        self.amount_of_values = amount_of_values
        pass

    def preprocess(self, x_input, y_input, n_poi):
        coefficients = self.calc_prominent_features(x_input, y_input, True)
        relevant_indices = np.argsort(-coefficients)[:n_poi][::-1]
        return x_input[:, relevant_indices]
     
    def calc_prominent_features(self, x_input, y_input, norm):
        # Group by label
        label_indices = []
        for i in range(self.amount_of_values):
            arr = np.nonzero(y_input == i)
            label_indices.append(arr)

        # Calculate mean and variance of each trace per label
        start = timer()
        input_length = len(x_input[0])
        coefficients = np.empty(input_length)
        
        for x in range(input_length):
            means = np.empty(self.amount_of_values)
            var = np.empty(self.amount_of_values)
            counts = np.empty(self.amount_of_values)
            result = 0

            for j in range(self.amount_of_values):
                values = x_input[label_indices[j], x]
                means[j] = np.mean(values)
                var[j] = np.var(values)
                counts[j] = len(values)
            
            for i in range(self.amount_of_values):
                for j in range(i):
                    temp = pow(means[i] - means[j], 2)
                    if norm:
                        temp /= (var[i]/counts[i] + var[j]/counts[j])
                    result += temp
            
            coefficients[x] = result
        end = timer()
        print("time taken: " + str(end-start) + " seconds")
        

        return coefficients

class DL_Preprocessor:
    
    def __init__(self) -> None:
        pass
    
    def preprocess(self, input):
        return input