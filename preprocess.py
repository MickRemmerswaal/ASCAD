from re import template
import numpy as np
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
    def __init__(self) -> None:
        pass

    def preprocess(self, x_input, y_input):
        indices = self.calc_prominent_features(x_input, y_input)

        return x_input[:indices]
     
    def calc_prominent_features(self, x_input, y_input):
        # Group by label
        start = timer()
        groups = []
        for i in range(5):
            groups.append([])
            for j in range(len(x_input)):
                if y_input[j] == i:
                    groups[i].append(j)
        end = timer()

        print("time taken: " + str((end - start)) + " seconds")
        
        start = timer()
        groups1 = []
        for i in range(256):
            arr = np.nonzero(y_input == i)
            arr = arr[0]
            groups1.append(arr)
        end = timer()

        print("time taken: " + str((end - start)) + " seconds")
        
        
        # Calculate mean and variance of each trace per label
        # Normalize if necessary

        return groups

class DL_Preprocessor:
    
    def __init__(self) -> None:
        pass
    
    def preprocess(self, input):
        return input