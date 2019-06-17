import numpy as np
from linear_classifier import LinearClassifier


class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = True
        self.smooth_param = 1
        
    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words    
        n_docs, n_words = x.shape
        
        # classes = a list of possible classes
        classes = np.unique(y)
        
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]
        
        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words,n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
            # prior[0] is the prior probability of a document being of class 0
            # likelihood[4, 0] is the likelihood of the fifth(*) feature being 
            # active, given that the document is of class 0
            # (*) recall that Python starts indices at 0, so an index of 4 
            # corresponds to the fifth feature!      
        # You need to incorporate self.smooth_param in likelihood calculation  
        ###########################


        # YOUR CODE HERE

        toks0 = np.zeros(n_words)  # total number of words in class pos
        print("Total number of words in class 0:")
        print(len(toks0))
        toks1 = np.zeros(n_words)  # total number of words in class neg
        print("Total number of words in class 1:")
        print(len(toks1))
        total_documents = np.zeros(n_classes)  # my positive and negative docs all together

        for i in range(n_docs):
            if y[i] == 0:  # if the doc is equal to zero, add it to my variable [0]
                total_documents[0] += 1
                for j in range(n_words): #looping the number of words in 0
                    toks0[j] += x[i, j]

            else:  # if not add it to variable [1]
                total_documents[1] += 1
                for j in range(n_words):  # looping the number of words in 1
                    toks1[j] += x[i, j]

        print("total number of documents is:") #print the total number of documents in both classes.
        print(n_docs)

        # Likelihood calculation:

        for i in range(n_words):
            likelihood[i][0] = (toks0[i] + self.smooth_param) / (toks0.sum() + self.smooth_param * n_words)
            likelihood[i][1] = (toks1[i] + self.smooth_param) / (toks1.sum() + self.smooth_param * n_words)

        # Prior calculation:

        prior[0] = total_documents[0] / n_docs
        prior[1] = total_documents[1] / n_docs

        ###########################

        params = np.zeros((n_words+1,n_classes))
        for i in range(n_classes): 
            # log probabilities
            params[0,i] = np.log(prior[i])
            with np.errstate(divide='ignore'): # ignore warnings
                params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
