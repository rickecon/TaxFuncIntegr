from __future__ import print_function
import ogusa
import os
import sys
import multiprocessing
from multiprocessing import Process
from dask.distributed import Client
from dask import compute, delayed
import dask.multiprocessing
import time
import numpy as np

from taxcalc import *
from ogusa.scripts.execute import runner
from ogusa.utils import REFORM_DIR, BASELINE_DIR


def run_ogusa(client, num_workers, tax_func_type, age_specific, guid,
              reform, user_params={}):
    '''
    Runs OG-USA model - both baseline and reform
    '''
    start_time = time.time()
    '''
    ------------------------------------------------------------------------
    Run Baseline First - this will be 2018 Law
    ------------------------------------------------------------------------
    '''
    output_base = os.path.join(BASELINE_DIR, guid)
    baseline_dir = os.path.join(BASELINE_DIR, guid)
    # Set corporate income tax rate
    user_params['tau_b'] = (0.21 * 0.55) * (0.017 / 0.055)
    kwargs = {'output_base': output_base, 'baseline_dir': baseline_dir,
              'test': False, 'time_path': True, 'baseline': True,
              'constant_rates': False, 'tax_func_type': tax_func_type,
              'analytical_mtrs': False, 'age_specific': age_specific,
              'user_params': user_params, 'guid': guid,
              'reform': {}, 'run_micro': True,
              'small_open': small_open, 'budget_balance': False,
              'baseline_spending': False, 'client': client,
              'num_workers': num_workers}
    runner(**kwargs)
    '''
    ------------------------------------------------------------------------
    Run Reform Second - this will be 2017 Law
    ------------------------------------------------------------------------
    '''
    output_base = os.path.join(REFORM_DIR, guid)
    # Set corporate income tax rate
    user_params['tau_b'] = (0.35 * 0.55) * (0.017 / 0.055)
    kwargs = {'output_base': output_base, 'baseline_dir': baseline_dir,
              'test': False, 'time_path': True, 'baseline': False,
              'constant_rates': False, 'tax_func_type': tax_func_type,
              'analytical_mtrs': False, 'age_specific': age_specific,
              'user_params': user_params, 'guid': guid,
              'reform': reform, 'run_micro': True,
              'small_open': small_open, 'budget_balance': False,
              'baseline_spending': False, 'client': client,
              'num_workers': num_workers}
    runner(**kwargs)
    print('run time = ', time.time()-start_time)


'''
Run several version of OG-USA with different tax functions
'''
# Define reform
# Note that TCJA is current law baseline in TC 0.16+
# Thus to compare TCJA to 2017 law, we'll use 2017 law as the reform but
# will use in "baseline" run below...
rec = Records()
pol = Policy()
calc = Calculator(policy=pol, records=rec)
ref = calc.read_json_param_objects('2017_law.json', None)
reform = ref['policy']

# Parameters constant across model runs
T_shifts = np.zeros(50)
T_shifts[2:10] = 0.01
T_shifts[10:40] = -0.01
G_shifts = np.zeros(6)
G_shifts[0:3] = -0.01
G_shifts[3:6] = -0.005
small_open = False
user_params = {'frisch': 0.41, 'start_year': 2018,
               'debt_ratio_ss': 1.0, 'T_shifts': T_shifts,
               'G_shifts': G_shifts, 'small_open': small_open}

# Define parameters to use for multiprocessing
# for local machine
client = Client(processes=False)
num_workers = multiprocessing.cpu_count()
# for cluster
# client = Client(scheduler_file='scheduler.json')
# num_workers = 7  # choose total number of workers when submit job,
# but here specify number of workers to use inside OG-USA
print('Number of workers = ', num_workers)

tax_func_list = ['DEP', 'DEP', 'DEP_totalinc', 'DEP_totalinc', 'GS',
                 'GS']
age_specific_list = [True, False, True, False, True, False]
guid_list = ['_DEP_Age', '_DEP_noAge', '_DEP_TI_Age', '_DEP_TI_noAge',
             '_GS_Age', '_GS_noAge']
# Create dictionary of lists with parameters of tax functions for each
# run, lists elements are: [tax_func_type, age_specific, guid]
run_dict = {'DEP_Age': ['DEP', True, '_DEP_Age'],
            'DEP_noAge': ['DEP', False, '_DEP_noAge'],
            'DEP_TI_Age': ['DEP_totalinc', True, '_DEP_TI_Age'],
            'DEP_TI_noAge': ['DEP_totalinc', False, '_DEP_TI_noAge'],
            'GS_Age': ['GS', True, '_GS_Age'],
            'GS_noAge': ['GS', False, '_GS_noAge']}


# Loop over different runs
for k, v in run_dict.items():
    print('Running model ', k)
    run_ogusa(client, num_workers, v[0], v[1], v[2], reform,
              user_params=user_params)
# lazy_values = []
# for k, v in run_dict.items():
#     lazy_values.append(
#         delayed(run_ogusa)(client, num_workers, v[0], v[1], v[2],
#                            reform, user_params=user_params))
# result = dask.delayed(print)(lazy_values)
# result.compute()
