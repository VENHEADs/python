
import yt.wrapper as yt
from nile.api.v1 import clusters
from nile.api.v1 import aggregators as na
from qb2.api.v1 import filters as qf
from nile.api.v1 import filters as nf
from nile.api.v1 import extractors as ne
import pandas as pd
import numpy as np
import timeit
import datetime
import os
from nile.api.v1 import statface as ns
from nile.api.v1 import cli
import ast
from nile.api.v1 import  Record
import re
import calendar


# In[ ]:


from yql.api.v1.client import YqlClient

def create_ab_table(date_start,date_end,experiment_name):
    client = YqlClient(db='',token='')

    request = client.query(
"""





""".format(date_start,date_end,experiment_name,experiment_name), syntax_version=1
        )
    request.run()
    for table in request.get_results():  # access to results blocks until they are ready
        while(table.fetch_full_data() == False):
            time.sleep(1)
            
    table = '//tmp/sovetnik/valeriy/{}'.format(experiment_name)
    return table

    
