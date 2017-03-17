'''
------------------------------------------------------------------------
Last updated 3/10/2017

This script produces the tables for the Tax Function Integration Paper

------------------------------------------------------------------------
'''

import pickle
import numpy as np


# Read in pickle of macro changes
# pickle.load(pct_changes, open("ogusa_output.pkl", "wb"))


# Read in tax function parameters
base_params =pickle.load(open('TxFuncEst_baselineint.pkl','rb'))
policy_params =pickle.load(open('TxFuncEst_policyint.pkl','rb'))

# create table showing variation in phi by age
table2 = np.zeros((3,4))
table2[0,:] = ((base_params['tfunc_etr_params_S'][:34,0,-1]).mean(),
               (base_params['tfunc_etr_params_S'][34:45,0,-1]).mean(),
               (base_params['tfunc_etr_params_S'][45:60,0,-1]).mean(),
               (base_params['tfunc_etr_params_S'][:,0,-1]).mean())
table2[1,:] = ((base_params['tfunc_mtrx_params_S'][:34,0,-1]).mean(),
               (base_params['tfunc_mtrx_params_S'][34:45,0,-1]).mean(),
               (base_params['tfunc_mtrx_params_S'][45:60,0,-1]).mean(),
               (base_params['tfunc_mtrx_params_S'][:,0,-1]).mean())
table2[2,:] = ((base_params['tfunc_mtry_params_S'][:34,0,-1]).mean(),
               (base_params['tfunc_mtry_params_S'][34:45,0,-1]).mean(),
               (base_params['tfunc_mtry_params_S'][45:60,0,-1]).mean(),
               (base_params['tfunc_mtry_params_S'][:,0,-1]).mean())
np.savetxt('table2.csv',table2,delimiter=',')


# create table showing parameter estimats from baseline and reform
table5 = np.zeros((14,6))
table5[:,0] = np.append(base_params['tfunc_etr_params_S'][21,0,:], [base_params['tfunc_etr_obs'][21,0], base_params['tfunc_etr_sumsq'][21,0]])
table5[:,1] = np.append(base_params['tfunc_mtrx_params_S'][21,0,:], [base_params['tfunc_mtrx_obs'][21,0], base_params['tfunc_mtrx_sumsq'][21,0]])
table5[:,2] = np.append(base_params['tfunc_mtry_params_S'][21,0,:], [base_params['tfunc_mtry_obs'][21,0], base_params['tfunc_mtry_sumsq'][21,0]])
table5[:,3] = np.append(policy_params['tfunc_etr_params_S'][21,0,:], [policy_params['tfunc_etr_obs'][21,0], policy_params['tfunc_etr_sumsq'][21,0]])
table5[:,4] = np.append(policy_params['tfunc_mtrx_params_S'][21,0,:], [policy_params['tfunc_mtrx_obs'][21,0], policy_params['tfunc_mtrx_sumsq'][21,0]])
table5[:,5] = np.append(policy_params['tfunc_mtry_params_S'][21,0,:], [policy_params['tfunc_mtry_obs'][21,0], policy_params['tfunc_mtry_sumsq'][21,0]]  )

np.savetxt('table5.csv',table5,delimiter=',')


# create table comparing our preferred specification with an alternative
base_alt_params =pickle.load(open('./ogusa/TxFuncEst_alt_baseline.pkl','rb'))
base_GS_params =pickle.load(open('./ogusa/TxFuncEst_GS_baseline.pkl','rb'))
#policy_alt_params =pickle.load(open('./ogusa/TxFuncEst_alt_policy.pkl','rb'))

table_sse = np.zeros((9,4))

table_sse[0,:] = np.append((base_params['tfunc_etr_sumsq'][:,0]).mean(),
               [(base_params['tfunc_etr_sumsq'][:34,0]).mean(),
               (base_params['tfunc_etr_sumsq'][34:45,0]).mean(),
               (base_params['tfunc_etr_sumsq'][45:60,0]).mean()])
table_sse[1,:] = np.append((base_params['tfunc_mtrx_sumsq'][:,0]).mean(),
               [(base_params['tfunc_mtrx_sumsq'][:34,0]).mean(),
               (base_params['tfunc_mtrx_sumsq'][34:45,0]).mean(),
               (base_params['tfunc_mtrx_sumsq'][45:60,0]).mean()])
table_sse[2,:] = np.append((base_params['tfunc_mtry_sumsq'][:,0]).mean(),
               [(base_params['tfunc_mtry_sumsq'][:34,0]).mean(),
               (base_params['tfunc_mtry_sumsq'][34:45,0]).mean(),
               (base_params['tfunc_mtry_sumsq'][45:60,0]).mean()])
table_sse[3,:] = np.append((base_alt_params['tfunc_etr_sumsq'][:,0]).mean(),
               [(base_alt_params['tfunc_etr_sumsq'][:34,0]).mean(),
               (base_alt_params['tfunc_etr_sumsq'][34:45,0]).mean(),
               (base_alt_params['tfunc_etr_sumsq'][45:60,0]).mean()])
table_sse[4,:] = np.append((base_alt_params['tfunc_mtrx_sumsq'][:,0]).mean(),
               [(base_alt_params['tfunc_mtrx_sumsq'][:34,0]).mean(),
               (base_alt_params['tfunc_mtrx_sumsq'][34:45,0]).mean(),
               (base_alt_params['tfunc_mtrx_sumsq'][45:60,0]).mean()])
table_sse[5,:] = np.append((base_alt_params['tfunc_mtry_sumsq'][:,0]).mean(),
               [(base_alt_params['tfunc_mtry_sumsq'][:34,0]).mean(),
               (base_alt_params['tfunc_mtry_sumsq'][34:45,0]).mean(),
               (base_alt_params['tfunc_mtry_sumsq'][45:60,0]).mean()])
table_sse[6,:] = np.append((base_GS_params['tfunc_etr_sumsq'][:,0]).mean(),
               [(base_GS_params['tfunc_etr_sumsq'][:34,0]).mean(),
               (base_GS_params['tfunc_etr_sumsq'][34:45,0]).mean(),
               (base_GS_params['tfunc_etr_sumsq'][45:60,0]).mean()])
table_sse[7,:] = np.append((base_GS_params['tfunc_mtrx_sumsq'][:,0]).mean(),
               [(base_GS_params['tfunc_mtrx_sumsq'][:34,0]).mean(),
               (base_GS_params['tfunc_mtrx_sumsq'][34:45,0]).mean(),
               (base_GS_params['tfunc_mtrx_sumsq'][45:60,0]).mean()])
table_sse[8,:] = np.append((base_GS_params['tfunc_mtry_sumsq'][:,0]).mean(),
               [(base_GS_params['tfunc_mtry_sumsq'][:34,0]).mean(),
               (base_GS_params['tfunc_mtry_sumsq'][34:45,0]).mean(),
               (base_GS_params['tfunc_mtry_sumsq'][45:60,0]).mean()])
np.savetxt('table_sse.csv',table_sse,delimiter=',')


table_se = np.zeros((9,4))

table_se[0,:] = np.append((base_params['tfunc_etr_sumsq'][:,0]).sum()/base_params['tfunc_etr_obs'][:,0].sum(),
               [(base_params['tfunc_etr_sumsq'][:34,0]).sum()/(base_params['tfunc_etr_obs'][:34,0]).sum(),
               (base_params['tfunc_etr_sumsq'][34:45,0]).sum()/(base_params['tfunc_etr_obs'][34:45,0]).sum(),
               (base_params['tfunc_etr_sumsq'][45:60,0]).sum()/(base_params['tfunc_etr_obs'][45:60,0]).sum()])
table_se[1,:] = np.append((base_params['tfunc_mtrx_sumsq'][:,0]).sum()/base_params['tfunc_mtrx_obs'][:,0].sum(),
               [(base_params['tfunc_mtrx_sumsq'][:34,0]).sum()/(base_params['tfunc_mtrx_obs'][:34,0]).sum(),
               (base_params['tfunc_mtrx_sumsq'][34:45,0]).sum()/(base_params['tfunc_mtrx_obs'][34:45,0]).sum(),
               (base_params['tfunc_mtrx_sumsq'][45:60,0]).sum()/(base_params['tfunc_mtrx_obs'][45:60,0]).sum()])
table_se[2,:] = np.append((base_params['tfunc_mtry_sumsq'][:,0]).sum()/base_params['tfunc_mtry_obs'][:,0].sum(),
               [(base_params['tfunc_mtry_sumsq'][:34,0]).sum()/(base_params['tfunc_mtry_obs'][:34,0]).sum(),
               (base_params['tfunc_mtry_sumsq'][34:45,0]).sum()/(base_params['tfunc_mtry_obs'][34:45,0]).sum(),
               (base_params['tfunc_mtry_sumsq'][45:60,0]).sum()/(base_params['tfunc_mtry_obs'][45:60,0]).sum()])
table_se[3,:] = np.append((base_alt_params['tfunc_etr_sumsq'][:,0]).sum()/base_alt_params['tfunc_etr_obs'][:,0].sum(),
               [(base_alt_params['tfunc_etr_sumsq'][:34,0]).sum()/(base_alt_params['tfunc_etr_obs'][:34,0]).sum(),
               (base_alt_params['tfunc_etr_sumsq'][34:45,0]).sum()/(base_alt_params['tfunc_etr_obs'][34:45,0]).sum(),
               (base_alt_params['tfunc_etr_sumsq'][45:60,0]).sum()/(base_alt_params['tfunc_etr_obs'][45:60,0]).sum()])
table_se[4,:] = np.append((base_alt_params['tfunc_mtrx_sumsq'][:,0]).sum()/base_alt_params['tfunc_mtrx_obs'][:,0].sum(),
               [(base_alt_params['tfunc_mtrx_sumsq'][:34,0]).sum()/(base_alt_params['tfunc_mtrx_obs'][:34,0]).sum(),
               (base_alt_params['tfunc_mtrx_sumsq'][34:45,0]).sum()/(base_alt_params['tfunc_mtrx_obs'][34:45,0]).sum(),
               (base_alt_params['tfunc_mtrx_sumsq'][45:60,0]).sum()/(base_alt_params['tfunc_mtrx_obs'][45:60,0]).sum()])
table_se[5,:] = np.append((base_alt_params['tfunc_mtry_sumsq'][:,0]).sum()/base_alt_params['tfunc_mtry_obs'][:,0].sum(),
               [(base_alt_params['tfunc_mtry_sumsq'][:34,0]).sum()/(base_alt_params['tfunc_mtry_obs'][:34,0]).sum(),
               (base_alt_params['tfunc_mtry_sumsq'][34:45,0]).sum()/(base_alt_params['tfunc_mtry_obs'][34:45,0]).sum(),
               (base_alt_params['tfunc_mtry_sumsq'][45:60,0]).sum()/(base_alt_params['tfunc_mtry_obs'][45:60,0]).sum()])
table_se[6,:] = np.append((base_GS_params['tfunc_etr_sumsq'][:,0]).sum()/base_GS_params['tfunc_etr_obs'][:,0].sum(),
               [(base_GS_params['tfunc_etr_sumsq'][:34,0]).sum()/(base_GS_params['tfunc_etr_obs'][:34,0]).sum(),
               (base_GS_params['tfunc_etr_sumsq'][34:45,0]).sum()/(base_GS_params['tfunc_etr_obs'][34:45,0]).sum(),
               (base_GS_params['tfunc_etr_sumsq'][45:60,0]).sum()/(base_GS_params['tfunc_etr_obs'][45:60,0]).sum()])
table_se[7,:] = np.append((base_GS_params['tfunc_mtrx_sumsq'][:,0]).sum()/base_GS_params['tfunc_mtrx_obs'][:,0].sum(),
               [(base_GS_params['tfunc_mtrx_sumsq'][:34,0]).sum()/(base_GS_params['tfunc_mtrx_obs'][:34,0]).sum(),
               (base_GS_params['tfunc_mtrx_sumsq'][34:45,0]).sum()/(base_GS_params['tfunc_mtrx_obs'][34:45,0]).sum(),
               (base_GS_params['tfunc_mtrx_sumsq'][45:60,0]).sum()/(base_GS_params['tfunc_mtrx_obs'][45:60,0]).sum()])
table_se[8,:] = np.append((base_GS_params['tfunc_mtry_sumsq'][:,0]).sum()/base_GS_params['tfunc_mtry_obs'][:,0].sum(),
               [(base_GS_params['tfunc_mtry_sumsq'][:34,0]).sum()/(base_GS_params['tfunc_mtry_obs'][:34,0]).sum(),
               (base_GS_params['tfunc_mtry_sumsq'][34:45,0]).sum()/(base_GS_params['tfunc_mtry_obs'][34:45,0]).sum(),
               (base_GS_params['tfunc_mtry_sumsq'][45:60,0]).sum()/(base_GS_params['tfunc_mtry_obs'][45:60,0]).sum()])
np.savetxt('table_se.csv',table_se,delimiter=',')



'''
FIGURES
'''

"""
This version: 2 Mar 2017
Written by Kerk Phillips

This program fits tax functions for "Integrating Microsimulation Models of Tax
Policy into a DGE Macroeconomic Model: A Canonical Example," by DeBaker, Evans
and Phillips.

This file fits data from Tax Calc to three different functions, compares the
goodness-of-fit, and plots the functions against the data.  The data sample
can be restricted by the age and the capital income of the individuals.
"""

import scipy.optimize as opt
import matplotlib.pyplot as plt

def GS(coeffs, *args):
    '''
    This is the functional from from Gouveia and Strass (1994) with an
    additional free parameter
    '''
    # unpack coefficients
    phi0, phi1, phi2 = coeffs
    # unpack data
    I, taxes, wgts = args
    # I = x+y
    errors = (taxes/I) - ((phi0*(I - (I**(-phi1) + phi2)**(-1/phi1)))/I)
    wsse = (wgts * (errors ** 2)).sum()
    print 'GS SSE: ', wsse
    print 'coeffs = ', phi0, phi1, phi2
    return wsse


'''
The functions below call those above and return the square-root of the sum of
squared errors (SSE) for each model
'''
# load data from pkl file - not this is not raw data from Tax Calc, but rather
# it is data with adjustments applied in txfunc.py, it's for the baseline in 2017
data_baseline  = pickle.load(open("./ogusa/cleaned_data.pkl", "rb"))

# calculate total taxes
data_baseline['Taxes'] = data_baseline['Effective Tax Rate']*data_baseline['Adjusted Total income']

# get time-series of interest  - for 43 year old in 2017
data_to_use = data_baseline[data_baseline['Age']==42].copy()
y = data_to_use['Total Capital Income'].values
x = data_to_use['Total Labor Income'].values
I = data_to_use['Adjusted Total income'].values
taxes = data_to_use['Taxes'].values
wgts = data_to_use['Weights'].values
etr_data = data_to_use['Effective Tax Rate'].values
tx_objs = (I, taxes, wgts)

# # set bounds on coefficients
# Gbounds = ((0, None), (0, None), (0, None))
#
# # set starting guesses for coefficients
# Gguess = np.array([0.3745, 0.7525, 0.7368])
#
# # use minimizer to solve for coefficients
# # Gouveia and Strauss
# Gout = opt.minimize(GS, Gguess,
#     args=(tx_objs), method="L-BFGS-B", bounds=Gbounds, tol=1e-15)
# Gcoeffs = Gout.x
# G_SSE = Gout.fun
# print 'GS coeffs, Obs, SSE: ', Gcoeffs, len(taxes), G_SSE


# Plot tax function curves in 2D
# plot GS vs alt function against data for a given year/age
npts = len(I)#100000       # points in grid
maxinc = 300000.  # upper end of grid
mininc = 0.  # lower end of grid
ygrid = np.zeros(npts)  # this is just a place holder to pass
xgrid = np.linspace(mininc, maxinc, npts)
datagrid = (ygrid, xgrid) # tuple to pass to functions
Gcoeffs = np.zeros((3,))
Gcoeffs[0] = (base_GS_params['tfunc_etr_params_S'][21,0,0])
Gcoeffs[1] = (base_GS_params['tfunc_etr_params_S'][21,0,1])
Gcoeffs[2] = (base_GS_params['tfunc_etr_params_S'][21,0,2])
Gyfit = (Gcoeffs[0]*(xgrid - (xgrid**(-Gcoeffs[1]) + Gcoeffs[2])**(-1/Gcoeffs[1])))/xgrid # put in terms of ETR by dividing by xgrid
A_alt = (base_alt_params['tfunc_etr_params_S'][21,0,0])
B_alt = (base_alt_params['tfunc_etr_params_S'][21,0,1])
max_rate_alt = (base_alt_params['tfunc_etr_params_S'][21,0,2])
min_rate_alt = (base_alt_params['tfunc_etr_params_S'][21,0,3])
Alt_fit = (((max_rate_alt - min_rate_alt) * ((A_alt * (xgrid**2) + B_alt * xgrid) /
    (A_alt * (xgrid**2) + B_alt * xgrid + 1))) + min_rate_alt)


A = (base_params['tfunc_etr_params_S'][21,0,0])
B = (base_params['tfunc_etr_params_S'][21,0,1])
C = (base_params['tfunc_etr_params_S'][21,0,2])
D = (base_params['tfunc_etr_params_S'][21,0,3])
max_x = (base_params['tfunc_etr_params_S'][21,0,4])
min_x = (base_params['tfunc_etr_params_S'][21,0,5])
max_y = (base_params['tfunc_etr_params_S'][21,0,6])
min_y = (base_params['tfunc_etr_params_S'][21,0,7])
shift_x = (base_params['tfunc_etr_params_S'][21,0,8])
shift_y = (base_params['tfunc_etr_params_S'][21,0,9])
shift = (base_params['tfunc_etr_params_S'][21,0,10])
share = (base_params['tfunc_etr_params_S'][21,0,11])
print 'Main spec params = ', base_params['tfunc_etr_params_S'][21,0,:]

X = xgrid.copy()
Y = xgrid.copy()*0.0
X2 = X ** 2
Y2 = Y ** 2
tau_x = (((max_x - min_x) * (A * X2 + B * X) / (A * X2 + B * X + 1))
    + min_x)
tau_y = (((max_y - min_y) * (C * Y2 + D * Y) / (C * Y2 + D * Y + 1))
    + min_y)
DEP_fit_nocapital = (((tau_x + shift_x) ** share) *
    ((tau_y + shift_y) ** (1 - share))) + shift

X = xgrid.copy()*0.7
Y = xgrid.copy()*0.3
X2 = X ** 2
Y2 = Y ** 2
tau_x = (((max_x - min_x) * (A * X2 + B * X) / (A * X2 + B * X + 1))
    + min_x)
tau_y = (((max_y - min_y) * (C * Y2 + D * Y) / (C * Y2 + D * Y + 1))
    + min_y)
DEP_fit_nolabor = (((tau_x + shift_x) ** share) *
    ((tau_y + shift_y) ** (1 - share))) + shift


# plot data
# plt.plot(I, etr_data, '.b', ms = 1, label='data')
plt.plot(I, etr_data, '.b', label='data')
plt.plot(xgrid, Gyfit, 'm-', lw=1, label='GS')
# plt.plot(xgrid, Alt_fit, 'r-', lw=1, label='ALT')
plt.plot(xgrid, DEP_fit_nocapital, '-', color='orange', lw=1, label='DEP, no y')
plt.plot(xgrid, DEP_fit_nolabor, '-',color='green', lw=1, label='DEP, mixed')
plt.xlabel('Total Income')
plt.ylabel('Effective Tax Rate')
# set axes range
plt.xlim(0, 100000)
plt.ylim(-0.2, 0.32)
plt.legend(loc='lower right')
plt.suptitle('Age = 43, Year = 2017')
# save high quality version to external file
plt.savefig('Compare_ETR_functions.png')
plt.show()

plt.plot(I, etr_data, '.b', label='data')
plt.plot(xgrid, DEP_fit_nocapital, '-', color='orange', lw=1, label='DEP, no y')
plt.plot(xgrid, DEP_fit_nolabor, '-',color='green', lw=1, label='DEP, mixed')
plt.xlabel('Total Income')
plt.ylabel('Effective Tax Rate')
# set axes range
plt.xlim(0, 100000)
plt.ylim(-0.2, 0.32)
plt.legend(loc='lower right')
plt.suptitle('Age = 43, Year = 2017')
# save high quality version to external file
plt.savefig('Compare_DEP_ETR_functions.png')
plt.show()
