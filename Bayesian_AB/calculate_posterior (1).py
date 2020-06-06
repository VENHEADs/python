import pandas as pd
import numpy as np
import timeit
import datetime
import os
from scipy.stats import beta, norm, uniform



def write_to_file(ab_name,text,metric_name):
    with open(str(ab_name), "a") as text_file:
        text_file.write(str(text) + ' for metric ' + metric_name+ '\n')


def make_decision(var_1,var_2,metric_name, posterior_dict_ctr, posterior_dict_convert, df_agg, \
                  ab_name = 'test',print_anyway = True):
    
    if metric_name == 'ctr':
        posterior_dict = posterior_dict_ctr
    if metric_name == 'convert':
        posterior_dict = posterior_dict_convert
        
    threshold_of_caring = 0.0001
    xgrid_size = 20000
    
    empirical_var_1_mean = df_agg.loc[var_1][metric_name]
    empirical_var_2_mean = df_agg.loc[var_2][metric_name]
    B_greater_A = empirical_var_2_mean > empirical_var_1_mean
    
    x = np.mgrid[0:xgrid_size,0:xgrid_size] / float(20*xgrid_size)
    pdf_arr = posterior_dict[var_1].pdf(x[0]) * posterior_dict[var_2].pdf(x[1])
    pdf_arr /= pdf_arr.sum() # normalization

    prob_error = np.zeros(shape=x[0].shape)
    if B_greater_A:
        prob_error[np.where(x[0] > x[1])] = 1.0
    else:
        prob_error[np.where(x[1] > x[0])] = 1.0

    expected_error = np.abs(x[0]-x[1])

    expected_err_scalar = (expected_error * prob_error * pdf_arr).sum()

    if (expected_err_scalar < threshold_of_caring) or print_anyway:
        if B_greater_A:
            
            line_1 = "Probability that version {} is larger than {} is ".format(var_2,var_1) \
                + str(((1-prob_error)*pdf_arr).sum())
            line_2 = " Expected error is " + str(expected_err_scalar)
                
            write_to_file(ab_name,line_1,metric_name)
            write_to_file(ab_name,line_2,metric_name)
        else:
            line_1 = "Probability that version {} is larger than {} is _ ".format(var_1,var_2) \
                + str(((1-prob_error)*pdf_arr).sum())
            line_2 = " Expected error is " + str(expected_err_scalar)
                
            write_to_file(ab_name,line_1,metric_name)
            write_to_file(ab_name,line_2,metric_name)
    else:
        print "Continue test. Expected error was " + str(expected_err_scalar) + " > " + str(threshold_of_caring)
    
    
