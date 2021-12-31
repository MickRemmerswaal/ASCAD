import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class PCA_Preprocessor:

    def __init__(self) -> None:
        pass
    
    def preprocess(self, variance, input):
        pca = PCA(variance)
        
        processed_input = pca.fit_transform(input)
        return processed_input

class LDA_Preprocessor:

    def __init__(self) -> None:
        pass

    def preprocess(self, x_input, y_input):
        lda = LinearDiscriminantAnalysis()
        
        processed_input = lda.fit_transform(x_input, y_input)
        return processed_input

class SOST_Preprocessor:

    def __init__(self) -> None:
        pass

    def preprocess(self, input):
        return input

class DL_Preprocessor:
    
    def __init__(self) -> None:
        pass
    
    def preprocess(self, input):
        return input