#====================================================================
# purpose: implementation of statistical tests in Python
# author: tirthankar chakravarty
# created: 2nd June 2014
# revised: 
# comments:
# 1. Make the one sample and two sample comparisons special cases of the 
#     K sample comparisons.
#    It seems to be simple to include one-sample as a special case of 
#     two-sample by checking if the second series is missing
#    Should be possible to specify *args as positional arguments that are 
#     the series that are to be compared.
#     TC: really need to see how comparable the forms of the K-sample and the 
#     2-sample tests are to make this decision
# 2. Make it possible to specify one-sided tests in all cases
#====================================================================
import numpy as np
import scipy.stats as sps
import warnings

# FIXME check that this is portable
# NOTE this does not get fixed when the name of the module is fixed
from .error_definitions import NotEqualLengthError

def test_mean(series, mean, variance = None, type = 'two-sided'):
    """ Test that the mean of the series is equal to given number.

    :param: series: a pandas Series
    :param: mean: the hypothesised true mean of the population
    :param: variance: if supplied, the known variance of the population
    
    :returns: a dict with the statistic and the p-value
    
    .. code-block:: python
        
        import pandas as pd
        import numpy as np
        # import unittest as ut
        
        from PyStatTests.tests import tests
        
        #================================================
        # test 1: Z-test for a population mean (variance known)
        # test 7: t-test for population mean (variance unknown)
        #================================================

        print("==============================================================")
        print("Tests of population mean under the null")
        print("==============================================================")
        # generate the univariate series
        series1 = pd.Series(np.random.randn(10))
        help(tests.test_means)
        # test 1: null  DGP
        print(tests.test_mean(series1, mean = 0, variance = 1))
        # test 7: null DGP
        print(tests.test_mean(series1, mean = 0))
        
        print("==============================================================")
        print("Tests of population mean under the alternative")
        print("==============================================================")
        # test 1: alternative DGP
        print(tests.test_mean(series1, mean = 2, variance = 1))
        # test 7: alternative DGP
        print(tests.test_mean(series1, mean = 2))

    """
    seriesMean = np.mean(series)
    N = series.count() 
    if variance is None: 
        # if the variance is not given, compute the variance
        # compute t-statistic, and compare to the t-distribution
        # NOTE the default variance computation does not deduct the
        #     the one DoF used up in the computation of the mean
        # http://stackoverflow.com/a/20708622/1414455
        variance = np.var(series, ddof = 1)
        statistic = (seriesMean - mean)/np.sqrt(variance/N)
        pvalue = 2*(1-sps.t.cdf(abs(statistic), N - 1))
    else:
        # if the variance is given, compute the z-statistic, and compare to 
        #     the normal distribution
        statistic = (seriesMean - mean)/np.sqrt(variance/N)
        pvalue = 2*(1 - sps.norm.cdf(abs(statistic)))
    
    return({'statistic': statistic, 'p-value': pvalue})
    
def test_means(series1, series2,  variance1 = None, variance2 = None, var_equal = False, 
          var_unknown = False, type = 'two-sided'):
    """Test whether two population means are different
    
    
    :param: series1, series 2: two pandas series from the two populations to be compared
    :param: variance1, variance2: if supplied, the known variances of the populations
    :param: var_equal: flag for if the two variances are equal
    :param: var_unknown: flag for whether the variances are unknown
    :param: type: whether to computer a two-sided, upper or lower tailed test
    
    :returns: a dict with the statistic, the p-value and the degrees of freedom of the test
    """
    mean1 = np.mean(series1)
    mean2 = np.mean(series2)
    
    N1 = series1.count()
    N2 = series2.count()
    dof = None
    
    if None in (variance1, variance2):
        var_unknown = True
        
        # NOTE if either of the variances is missing, then both are recomputed,
        variance1 = np.var(series1, ddof = 1)
        variance2 = np.var(series2, ddof =1)
    elif variance1 == variance2:
        var_equal = True
    
    if not var_unknown and var_equal:
        # test 2: Z-test for two population means (variances known and equal)
        statistic = (mean1 - mean2)/np.sqrt(variance1*(1/N1+1/N2))
        pvalue = 2*(1-sps.norm.cdf(abs(statistic)))
    elif not var_unknown and not var_equal:
        # test 3: Z-test for two population means (variances known and not equal)
        statistic = (mean1 - mean2)/np.sqrt((variance1/N1 + variance2/N2))
        pvalue = 2*(1-sps.norm.cdf(abs(statistic)))
    elif var_equal and var_unknown:
        # test 8: t-test for two population means (variances unknown and equal)
        # compute the unified equal variance
        variance = ((N1 - 1)*variance1 + (N2 - 1)*variance2)/(N1 + N2 - 2)
        statistic = (mean1 - mean2)/np.sqrt(variance*(1/N1+1/N2))
        dof = N1 + N2 - 2
        pvalue = 2*(1-sps.t.cdf(abs(statistic), dof))
    else:
        # test 9: t-test for two population means (variances unknown and not equal)
        # note that this is the most general case where the variances are not
        #    known, and the variances are not assumed to be equal. This should
        #    be the sink case.
        statistic = (mean1 - mean2)/np.sqrt((variance1/N1 + variance2/N2))
        dof = ((variance1/N1 + variance2/N2)**2/(variance1**2/(N1**2 * (N1-1)) + variance2**2/(N2**2 * (N2-1))))
        pvalue = 2*(1-sps.t.cdf(abs(statistic), dof))
    
    return({'statistic': statistic, 'p-value': pvalue, 'DoF': dof})

def test_variance(series, variance):
    """Test whether the population variance is equal to a given value
    
    :param: series: sample of data from a population
    :param: variance: hypothesised variance of the population
    
    :returns: a dict containing the statistic, p-value and the degrees of freedom of the test
    """
    sample_variance = np.var(series, ddof = 1)
    N = series.count()
    statistic = (N-1)*(sample_variance/variance)
    dof = N - 1
    pvalue = 1-sps.chi2.cdf(statistic, dof) 
    return({'statistic': statistic, 'p-value': pvalue,
             'DoF': dof})
    
def test_proportion(series, proportion):
    """Test whether the population proportion is equal to a given value
    
    :param: series: sample of data from a population
    :param: proportion: hypothesised proportion of the binomial population
    
    :returns: a dict containing the test statistic and the p-value of the test
    """
    N = series.count()
    sample_proportion = np.mean(series)
    statistic = ((abs(proportion - sample_proportion) - 0.5/N)/
                 np.sqrt(proportion*(1 - proportion)/N))
    pvalue = 2*(1 - sps.norm.cdf(statistic))
    return({'statistic': statistic, 'p-value': pvalue})



def test_correlation(series1, series2, correlation, test_type = 'z'):
    """Test population correlation is equal to a given value
    if the correlation passed is zero, then a t-test is called
    else, a z-test is called. Actually it is possible to also compute the
    z-test when the hypothesised population correlation is zero.
    
    :param: series1, series2: the pandas series whose correlation has to be computed
    
    :returns: the test statistic, the p-value, the degrees of freedom, and the type of the test
    """
    N = series1.count()
    sample_correlation = series1.corr(series2)
    dof = None
    
    try:
        assert(series1.count() == series2.count()) 
    except AssertionError:
        raise(NotEqualLengthError('The series are not of equal length.')) from None
    
    # change the type of the test if 't' and the correlation is not zero    
    if test_type == 't':
        try:
            assert(correlation == 0)
        except AssertionError: 
            warnings.warn('A t-test cannot be computed with a non-zero '
                          ' hypothesised correlation coefficient, switching to a z-test.')
            test_type  = 'z'
    
    if test_type  == 'z':
        # compute the z-test
        sample_mean = 0.5*np.log((1 + sample_correlation)/(1 - sample_correlation))
        population_mean = 0.5*np.log((1 + correlation)/(1 - correlation))
        population_std = 1/np.sqrt(N - 3)
        statistic = (sample_mean - population_mean)/population_std
        pvalue = 2*(1-sps.norm.cdf(abs(statistic)))
    elif test_type == 't':
        # compute the t-test
        statistic = np.sqrt(N-2)*sample_correlation/np.sqrt(1 - sample_correlation**2)
        dof = N - 2
        pvalue = 2*(1-sps.t.cdf(abs(statistic), dof))
    
    return({'statistic': statistic, 'p-value': pvalue, 'DoF': dof, 'test_type': test_type}) 

def test_proportions(series1, series2):
    """ Test the significance of the difference between two proportions
    
    :param: series1, series2: pandas series from which the proportion is to be computed
    
    :returns: a dict with the statistic and the p-value
    """
    N1 = series1.count()
    N2 = series2.count()
    
    # check that the length of the two series are the same
    # TODO this seems to go against DRY. Need to wrap it in a class which
    #     accepts two series and does the necessary checks that are not 
    #     test dependent, before implementing the test
    try:
        assert(N1 == N2)
    except AssertionError:
        raise NotEqualLengthError from None
    
    # TODO check that it is indeed proportion data. is this overly zealous?
    # TODO add handling of missing data to all of this
    
    mean1 = np.mean(series1)
    mean2 = np.mean(series2)
    
    combined_mean = (mean1*N1 + mean2*N2)/(N1 + N2)
    
    statistic = ((mean1 - mean2)/
                 np.sqrt((combined_mean*(1-combined_mean)/(1/N1 + 1/N2))))
    pvalue = 2*(1-sps.norm.cdf(abs(statistic)))
    
    return({'statistic': statistic, 'p-value': pvalue})
