#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def calculate_csi(expected, actual, buckettype='bins', buckets=10, axis=0):
    csi_values = []
    for col in expected:
        train_series = expected[col]
        test_series = actual[col]
        csi = np.round(calculate_psi(train_series, test_series, buckettype='bins', buckets=10, axis=1,is_psi=False),5)
        csi_values.append(csi)
    result = pd.DataFrame(    {'Feature': expected.columns.tolist(),
         'Characteristic Stability Index (CSI)': csi_values
        })
    return result
    
        
    
def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0,is_psi=True):
    if is_psi:
        cols = expected.columns.tolist()
    expected = expected.to_numpy()
    actual = actual.to_numpy()
    
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)
    if is_psi:
        psi_values = list(psi_values)

        result = pd.DataFrame(    {'Feature': cols,
         'Population Stability Index (PSI):': psi_values
        })
        return(result)
    else:
        return psi_values


# In[2]:


train_data = pd.read_csv('traindataSIT.csv', error_bad_lines=False, index_col=0)
test_data = pd.read_csv('testdataSIT.csv', error_bad_lines=False, index_col=0)
csi = np.round(calculate_csi(train_data,test_data, buckettype='bins', buckets=10, axis=1),5)
psi = np.round(calculate_psi(train_data,test_data, buckettype='bins', buckets=10, axis=1),5)
csi.set_index('Feature', inplace=True)
psi.set_index('Feature', inplace=True)
data = csi.join(psi)
data.to_csv("PSI_CSI_Matrics.csv")


# In[115]:


data


# In[ ]:




