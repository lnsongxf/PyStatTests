'''
# purpose: implement tests using both NumPy and pandas datasets explicitly
#     even though using pandas guarantees use with NumPy
# author: tirthankar chakravarty
# created: 
# revised:
# comments:
'''

import pandas as pd
import numpy as np
# import unittest as ut

from stattests.tests import tests

#================================================
# test 1: Z-test for a population mean (variance known)
# test 7: t-test for population mean (variance unknown)
#================================================

print("==============================================================")
print("Tests of population mean under the null")
print("==============================================================")
# generate the univariate series
series1 = pd.Series(np.random.randn(10))
help(tests.test2)
# test 1: null  DGP
print(tests.test1(series1, mean = 0, variance = 1))
# test 7: null DGP
print(tests.test1(series1, mean = 0))

print("==============================================================")
print("Tests of population mean under the alternative")
print("==============================================================")
# test 1: alternative DGP
print(tests.test1(series1, mean = 2, variance = 1))
# test 7: alternative DGP
print(tests.test1(series1, mean = 2))

#================================================
# test 2: Z-test for two population means (variances known and equal)
# test 3: Z-test for two population means (variances known and unequal)
# test 8: t-test for two population means (variances unknown and equal)
# test 9: t-test for two population means (variances unknown and unequal)
#================================================
print("==============================================================")
print("Tests of two population means under the null")
print("==============================================================")
help(tests.test2)

# test2 Null: variances equal and 4
series1 = pd.Series(2*np.random.randn(10))
series2 = pd.Series(2*np.random.randn(10))
print(tests.test2(series1, series2, variance1 = 4, variance2 = 4))

# test3 Null: variances unequal 
series1 = pd.Series(2*np.random.randn(10))
series2 = pd.Series(3*np.random.randn(10))
print(tests.test2(series1, series2, variance1 = 4, variance2 = 6))

# test8 Null: variances equal and unknown
series1 = pd.Series(33*np.random.randn(10))
series2 = pd.Series(33*np.random.randn(10))
print(tests.test2(series1, series2))

# test9 Null: variances equal and unknown
series1 = pd.Series(19*np.random.randn(10))
series2 = pd.Series(26*np.random.randn(10))
print(tests.test2(series1, series2))

print("==============================================================")
print("Tests of two population means under the alternative")
print("==============================================================")
# test2 Alternative: variances equal and 4
series1 = pd.Series(3 + 2*np.random.randn(10))
series2 = pd.Series(6 + 2*np.random.randn(10))
print(tests.test2(series1, series2, variance1 = 4, variance2 = 4))

# test3 Alternative: variances unequal 
series1 = pd.Series(9 + 2*np.random.randn(10))
series2 = pd.Series(11 + 3*np.random.randn(10))
print(tests.test2(series1, series2, variance1 = 4, variance2 = 6))

# test8 Alternative: variances equal and unknown
series1 = pd.Series(14 + 33*np.random.randn(10))
series2 = pd.Series(-22 + 33*np.random.randn(10))
print(tests.test2(series1, series2))

# test9 Alternative: variances equal and unknown
series1 = pd.Series(88 + 19*np.random.randn(10))
series2 = pd.Series(86 + 26*np.random.randn(10))
print(tests.test2(series1, series2))
# NOTE the small sample size bites in this case; low power of the test

print("==============================================================")
print("Tests of population variance under the null")
print("==============================================================")
help(tests.test3)
series1 = pd.Series(13*np.random.randn(10))
print(tests.test3(series1, variance = 13**2))

print("==============================================================")
print("Tests of population variance under the alternative")
print("==============================================================")
print(tests.test3(series1, variance = 12**2))
# NOTE again the small sample size bites; unable to reject in most cases


print("==============================================================")
print("Tests of population proportion under the null")
print("==============================================================")
help(tests.test4)
series1 = pd.Series(np.random.binomial(n = 1, p = 0.7, size = 10))
print(tests.test4(series1, proportion = 0.7))

print("==============================================================")
print("Tests of population proportion under the alternative")
print("==============================================================")
series1 = pd.Series(np.random.binomial(n = 1, p = 0.9, size = 10))
print(tests.test4(series1, proportion = 0.8))
