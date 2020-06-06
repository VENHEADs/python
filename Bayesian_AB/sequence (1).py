import yt.wrapper as yt
import pandas as pd
import numpy as np
import timeit
import datetime
import os
from scipy.stats import beta, norm, uniform
from nile.api.v1 import statface as ns
from nile.api.v1 import cli
import ast
import re
from multiprocessing import Pool
from scipy.stats import ttest_ind,mannwhitneyu,kstest,normaltest
from yql.api.v1.client import YqlClient
from make_table import create_ab_table
from calculate_posterior import make_decision
import subprocess

def analyze_sequential_test(ab_name, date_start, date_end):
    
    try:
        print subprocess.check_output(['rm',ab_name])
        print subprocess.check_output(['rm','{}.xlsx'.format(ab_name)])
    except:
        pass
        
    client = yt.YtClient("",token='')
#     create_ab_table(date_start,date_end,ab_name)
    table = create_ab_table(date_start,date_end,ab_name)
    
    x = client.read_table(table) 
    results = []
    for row in x:
        results.append(row)
    df = pd.DataFrame(results)
    df_agg = df.groupby('ab').sum()
    df_agg['ctr'] = df_agg.clicks/df_agg.shows
    df_agg['convert'] = df_agg.buys/df_agg.clicks
    
    list_of_variants = df_agg.index.tolist() # posteriors list
    posterior_dict_convert = {}
    posterior_dict_ctr = {}
    for var in list_of_variants:
        posterior_dict_convert[var] = beta(249 + df_agg.loc[var].buys,  \
                                           14269 - df_agg.loc[var].buys + df_agg.loc[var].clicks)

        posterior_dict_ctr[var] = beta(110 + df_agg.loc[var].clicks,  \
                                           5550 - df_agg.loc[var].clicks + df_agg.loc[var].shows)

    for metric in ['ctr','convert']:
        for variant in list_of_variants:
            make_decision(variant,'control',metric,posterior_dict_ctr, posterior_dict_convert, df_agg, \
                          ab_name)
    df_agg.to_excel('{}.xlsx'.format(ab_name))
    
