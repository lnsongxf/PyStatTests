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
import scipy as sp
import scipy.stats as sps

def test1(series, mean, variance = None, type = 'two-sided'):
    """ Test that the mean of the series is equal to given number.
    
    Keyword arguments:
    series: a pandas Series
    mean: the hypothesised true mean of the population
    variance: if supplied, the known variance of the population
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
    
def test2(series1, series2,  variance1 = None, variance2 = None, var_equal = False, 
          var_unknown = False, type = 'two-sided'):
    """Test whether two population means are different
    
    Keyword arguments:
    series1, series 2: two pandas series from the two populations to be compared
    variance1, variance2: if supplied, the known variances of the populations
    var_equal: flag for if the two variances are equal
    var_unknown: flag for whether the variances are unknown
    type: whether to computer a two-sided, upper or lower tailed test
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

def test3(series, variance):
    """Test whether the population variance is equal to a given value
    
    Keyword arguments:
    series: sample of data from a population
    variance: hypothesised variance of the population
    """
    sample_variance = np.var(series, ddof = 1)
    N = series.count()
    statistic = (N-1)*(sample_variance/variance)
    dof = N - 1
    pvalue = 1-sps.chi2.cdf(statistic, dof) 
    return({'statistic': statistic, 'p-value': pvalue,
             'DoF': dof})
    
def test4(series, proportion):
    """Test whether the population proportion is equal to a given value
    
    Keyword arguments:
    series: sample of data from a population
    proportion: hypothesised proportion of the binomial population
    """
    N = series.count()
    sample_proportion = np.mean(series)
    statistic = ((abs(proportion - sample_proportion) - 0.5/N)/
                 np.sqrt(proportion*(1 - proportion)/N))
    pvalue = 2*(1 - sps.norm.cdf(statistic))
    return({'statistic': statistic, 'p-value': pvalue})
    