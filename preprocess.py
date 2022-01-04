import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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

    def preprocess(self, n_features, input):



        return input

    def select_top_n(self, n_features, input):

        return 
        
class DL_Preprocessor:
    
    def __init__(self) -> None:
        pass
    
    def preprocess(self, input):
        return input