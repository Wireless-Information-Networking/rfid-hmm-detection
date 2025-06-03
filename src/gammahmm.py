# Class to implement a Hidden Markov Model with Gamma emission distribution

import numpy as np
from hmmlearn.base import BaseHMM
from hmmlearn.utils import normalize
import scipy.stats as stats

class GammaHMM(BaseHMM):
    """
    Hidden Markov Model with Gamma emission distribution.
    
    Parameters
    ----------
    n_components : int
        Number of states in the model.
    
    n_iter : int
        Maximum number of iterations to perform.
    
    tol : float
        Convergence threshold.
    
    params : string
        Controls which parameters are updated in the training process.
        's' for start probability.
        't' for transition matrix.
        'h' for shape of the Gamma distribution.
        'l' for localization to center the distribution.
        'c' for scale to adapt the distribution.

    init_params : string
        Controls which parameters are initialized prior to training.
        's' for start probability.
        't' for transition matrix.
        'h' for shape of the Gamma distribution.
        'l' for localization to center the distribution.
        'c' for scale to adapt the distribution.

    random_state : int
        Seed for random number generator.

    verbose : bool
        If True, print progress messages.

    implementation : string
        Implementation method to use.

    algorithm : string
        Decoding algorithm to use.

    shape : array
        Shape parameter for the Gamma distribution.
    
    localization : array
        Localization parameter in order to adapt the Gamma distribution.
    
    scale : array
        Scale parameter for adapt the Gamma distribution.
    
    Attributes
    ----------
    n_features : int
        Number of features.

    n_params : int
        Number of parameters.

    transmat_ : numpy array
        Transition matrix.

    startprob_ : numpy array
        Initial state occupation distribution.

    shape : numpy array
        Shape parameter for the Gamma distribution.
    
    localization : numpy array
        Localization parameter for the Gamma distribution.
    
    scale : numpy array
        Scale parameter for the Gamma distribution.
    """
    def __init__(self, n_components=1, 
                 n_iter=10, tol=1e-2, 
                 params='sthcl', init_params='sthcl',
                 random_state=None, 
                 verbose=False,
                 implementation='log',
                 algorithm='viterbi',
                 shape = None,
                 localization = None,
                 scale = None):
        super().__init__(n_components,
                            n_iter=n_iter, 
                            tol=tol,
                            params=params, 
                            init_params=init_params,
                            random_state=random_state,
                            verbose=verbose, 
                            implementation=implementation, 
                            algorithm=algorithm)
                
        self.shape = shape
        self.localization = localization
        self.scale = scale

    def _init(self, X, lengths=None):
        super()._init(X, lengths)
        
        # start probability distributions
        if 'h' in self.init_params:
            self.shape = np.ones(self.n_components)
        if 'l' in self.init_params:
            self.localization = np.zeros(self.n_components)
        if 'c' in self.init_params:
            self.scale = np.ones(self.n_components)


    
    def _check(self):
        super()._check()

        # check for params of the Gamma distribution
        self.__shape = np.asarray(self.shape)
        self.__localization = np.asarray(self.localization)
        self.__scale = np.asarray(self.scale)

        if self.__shape.shape != (self.n_components, self.n_features):
            raise ValueError("shape must have shape (n_components, n_features)")
        if self.__localization.shape != (self.n_components, self.n_features):
            raise ValueError("localization must have shape (n_components, n_features)")
        if self.__scale.shape != (self.n_components, self.n_features):
            raise ValueError("scale must have shape (n_components, n_features)")
        

    def _compute_log_likelihood(self, X):
        
        return stats.gamma.logpdf(X, df=self.df)

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        return stats
    
    def _accumulate_sufficient_statistics(self, stats, X, framelogprob, posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        stats['post'] += posteriors.sum(axis=0)
        return stats
    
    def _do_mstep(self, stats):
        super()._do_mstep(stats)

