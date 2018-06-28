'''
------------------------------------------------------------------------
Last updated 4/19/2018

This script produces the tables and figures for the Tax Function
Integration Paper

------------------------------------------------------------------------
'''

import pickle
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import os
import xlsxwriter
import ogusa
from ogusa.utils import REFORM_DIR, BASELINE_DIR

# Read in tax function parameters
tax_func_params = {}
tax_func_list = ['DEP', 'DEP', 'DEP_totalinc', 'DEP_totalinc', 'GS',
                 'GS']
age_specific_list = [True, False, True, False, True, False]
guid_list = ['_DEP_TI_noAge', '_DEP_TI_Age', '_DEP_noAge', '_DEP_Age',
             '_GS_noAge', '_GS_Age']
tax_func_params_base = {}
tax_func_params_reform = {}
tpi_base = {}
tpi_reform = {}
ss_base = {}
ss_reform = {}
for guid in guid_list:
        # NOTE TAHT SWITHC BASELINE AND REFORM SINCE TCJA CURRENT LAW
        reform_path = os.path.join(BASELINE_DIR, guid,
                                   'TxFuncEst_baseline' + guid + '.pkl')
        base_path = os.path.join(REFORM_DIR, guid,
                                 'TxFuncEst_policy' + guid + '.pkl')
        tax_func_params_base[guid] = pickle.load(
            open(base_path, 'rb'), encoding='latin')
        tax_func_params_reform[guid] = pickle.load(
            open(reform_path, 'rb'), encoding='latin')
        tpi_reform_path = os.path.join(BASELINE_DIR, guid,
                                       'TPI/TPI_vars.pkl')
        tpi_base_path = os.path.join(REFORM_DIR, guid,
                                     'TPI/TPI_vars.pkl')
        tpi_base[guid] = pickle.load(
            open(tpi_base_path, 'rb'), encoding='latin')
        tpi_reform[guid] = pickle.load(
            open(tpi_reform_path, 'rb'), encoding='latin')
        ss_reform_path = os.path.join(BASELINE_DIR, guid,
                                      'SS/SS_vars.pkl')
        ss_base_path = os.path.join(REFORM_DIR, guid,
                                    'SS/SS_vars.pkl')
        ss_base[guid] = pickle.load(
            open(ss_base_path, 'rb'), encoding='latin')
        ss_reform[guid] = pickle.load(
            open(ss_reform_path, 'rb'), encoding='latin')

'''
------------------------------------------------------------------------
    Tables - all tables for paper saved to different worksheets in an
    Excel workbook
------------------------------------------------------------------------
'''
# open Excel workbook
workbook = xlsxwriter.Workbook('TFI_Tables.xlsx',
                               {'nan_inf_to_errors': True})

# Table 2: variation in phi by age for DEP function
# create list of list with table info
rate_labels = ['$ETR$', '$MTRx$', '$MTRy$']
rate_types = ['etr', 'mtrx', 'mtry']
table2 = []
table2.append([''])
table2[0].extend(('21 to 54', '55 to 65', '66 to 80', 'All ages'))
for i, label in enumerate(rate_labels):
    table2.append([label])
    table2[i + 1].extend((
        tax_func_params_base['_DEP_Age']['tfunc_' + rate_types[i] +
                                         '_params_S'][:34, 0, -1].mean(),
        tax_func_params_base['_DEP_Age']['tfunc_' + rate_types[i] +
                                         '_params_S'][34:45, 0, -1].mean(),
        tax_func_params_base['_DEP_Age']['tfunc_' + rate_types[i] +
                                         '_params_S'][45:60, 0, -1].mean(),
        tax_func_params_base['_DEP_Age']['tfunc_' + rate_types[i] +
                                         '_params_S'][:, 0, -1].mean()))
# save table of info to Excel
worksheet = workbook.add_worksheet('Table 2')
worksheet.merge_range('B1:D1', 'Age ranges')
for i, val in enumerate(table2):
    for j, val2 in enumerate(table2[i]):
        worksheet.write(i + 1, j, val2)

# Table 3: comparing std errors across tax functions
tax_func_labels = ['Ratio of polynomials, ETR',
                   'Ratio of polynomials, vary by age, ETR',
                   'Ratio of polynomials, vary by income source ETR',
                   'Ratio of polynomials, vary by age and income source ETR',
                   'Gouveia and Strauss (1994), ETR',
                   'Gouveia and Strauss (1994), vary by age, ETR']
table3 = []
table3.append([''])
table3[0].extend(('All ages', '21 to 54', '55 to 65', '66 to 80'))
for i, label in enumerate(tax_func_labels):
    table3.append([label])
    table3[i + 1].extend((
        tax_func_params_base[guid_list[i]]['tfunc_etr_sumsq'][:, 0].sum()
        / tax_func_params_base[guid_list[i]]['tfunc_etr_obs'][:, 0].sum(),
        tax_func_params_base[guid_list[i]]['tfunc_etr_sumsq'][:34, 0].sum()
        / tax_func_params_base[guid_list[i]]['tfunc_etr_obs'][:34, 0].sum(),
        tax_func_params_base[guid_list[i]]['tfunc_etr_sumsq'][34:45, 0].sum()
        / tax_func_params_base[guid_list[i]]['tfunc_etr_obs'][34:45, 0].sum(),
        tax_func_params_base[guid_list[i]]['tfunc_etr_sumsq'][45:60, 0].sum()
        / tax_func_params_base[guid_list[i]]['tfunc_etr_obs'][45:60, 0].sum()))
# save table of info to Excel
worksheet = workbook.add_worksheet('Table 3')
worksheet.merge_range('B1:E1', 'Age ranges')
for i, val in enumerate(table3):
    for j, val2 in enumerate(table3[i]):
        worksheet.write(i + 1, j, val2)

# Table 4: comparing SSE across tax functions
table4 = []
table4.append([''])
table4[0].extend(('All ages', '21 to 54', '55 to 65', '66 to 80'))
for i, label in enumerate(tax_func_labels):
    table4.append([label])
    table4[i + 1].extend((
        tax_func_params_base[guid_list[i]]['tfunc_etr_sumsq'][:, 0].mean(),
        tax_func_params_base[guid_list[i]]['tfunc_etr_sumsq'][:34, 0].mean(),
        tax_func_params_base[guid_list[i]]['tfunc_etr_sumsq'][34:45, 0].mean(),
        tax_func_params_base[guid_list[i]]['tfunc_etr_sumsq'][45:60, 0].mean()))
# save table of info to Excel
worksheet = workbook.add_worksheet('Table 4')
worksheet.merge_range('B1:E1', 'Age ranges')
for i, val in enumerate(table4):
    for j, val2 in enumerate(table4[i]):
        worksheet.write(i + 1, j, val2)

# Table 5: comparing observations across tax functions
table5 = []
table5.append([''])
table5[0].extend(('All ages', '21 to 54', '55 to 65', '66 to 80'))
for i, label in enumerate(tax_func_labels):
    table5.append([label])
    table5[i + 1].extend((
        tax_func_params_base[guid_list[i]]['tfunc_etr_obs'][:, 0].sum(),
        tax_func_params_base[guid_list[i]]['tfunc_etr_obs'][:34, 0].sum(),
        tax_func_params_base[guid_list[i]]['tfunc_etr_obs'][34:45, 0].sum(),
        tax_func_params_base[guid_list[i]]['tfunc_etr_obs'][45:60, 0].sum()))
# save table of info to Excel
worksheet = workbook.add_worksheet('Table 5')
worksheet.merge_range('B1:E1', 'Age ranges')
for i, val in enumerate(table5):
    for j, val2 in enumerate(table5[i]):
        worksheet.write(i + 1, j, val2)

# Table 6: parameter estimates from baseline and reform for DEP funcs
# report parameters from function for 42 year old in first year of window
param_names = ['$A$', '$B$', '$C$', '$D$', '$max_x$', '$min_x$',
               '$max_y$', '$min_y$', '$shift_x$', '$shift_y$',
               '$shift$', '$share$']
table6 = []
table6.append(['Parameter'])
table6[0].extend(tuple(rate_labels * 2))
for i, label in enumerate(param_names):
    table6.append([label])
    for j, rate in enumerate(rate_types):  # for baseline results
        table6[i + 1].append(
            tax_func_params_base['_DEP_Age']['tfunc_' + rate +
                                             '_params_S'][21, 0, i])
    for j, rate in enumerate(rate_types):  # for reform results
        table6[i + 1].append(
            tax_func_params_reform['_DEP_Age']['tfunc_' + rate +
                                               '_params_S'][21, 0, i])
table6.append(['Obs (N)'])
table6.append(['SSE'])
for j, rate in enumerate(rate_types):
    table6[-2].append(
        tax_func_params_base['_DEP_Age']['tfunc_' + rate +
                                         '_obs'][21, 0])
    table6[-1].append(
        tax_func_params_base['_DEP_Age']['tfunc_' + rate +
                                         '_sumsq'][21, 0])
for j, rate in enumerate(rate_types):
    table6[-2].append(
        tax_func_params_reform['_DEP_Age']['tfunc_' + rate +
                                           '_obs'][21, 0])
    table6[-1].append(
        tax_func_params_reform['_DEP_Age']['tfunc_' + rate +
                                           '_sumsq'][21, 0])
# save table of info to Excel
worksheet = workbook.add_worksheet('Table 6')
worksheet.merge_range('B1:D1', '2017 Law')
worksheet.merge_range('E1:G1', 'TCJA')
for i, val in enumerate(table6):
    for j, val2 in enumerate(table6[i]):
        worksheet.write(i + 1, j, val2)


# Table 8: GDP changes across tax functions
tax_func_labels = ['Ratio of polynomials, ETR',
                   'Ratio of polynomials, vary by age, ETR',
                   'Ratio of polynomials, vary by income source ETR',
                   'Ratio of polynomials, vary by age and income source ETR',
                   'Gouveia and Strauss (1994), ETR',
                   'Gouveia and Strauss (1994), vary by age, ETR']
table8 = []
for i, label in enumerate(tax_func_labels):
    table8.append([label])
    for y in range(10):
        table8[i].append(((tpi_reform[guid_list[i]]['Y'][y] -
                           tpi_base[guid_list[i]]['Y'][y])
                          / tpi_base[guid_list[i]]['Y'][y]))
    table8[i].extend((
        ((tpi_reform[guid_list[i]]['Y'][:11].sum() -
          tpi_base[guid_list[i]]['Y'][:11].sum())
         / tpi_base[guid_list[i]]['Y'][:11].sum()),
        ((ss_reform[guid_list[i]]['Yss'] -
          ss_base[guid_list[i]]['Yss'])
         / ss_base[guid_list[i]]['Yss'])))
# save table of info to Excel
worksheet = workbook.add_worksheet('Table 8')
worksheet.write(0, 0, 'Tax Function')
for y in range(2018, 2028):
    worksheet.write(0, y - 2017, str(y))
worksheet.write(0, 11, '2018-2027')
worksheet.write(0, 12, 'SS')
for i, val in enumerate(table8):
    for j, val2 in enumerate(table8[i]):
        worksheet.write(i + 1, j, val2)

# Table 9: Changes in all macro aggregates for DEP only
results_labels = ['GDP', 'Conusmption', 'Investment', 'Hours Worked',
                  'Avg. Wage', 'Interest Rate', 'Total Taxes']
var_names = ['Y', 'C', 'I', 'L', 'w', 'r', 'REVENUE']
ss_var_names = ['Yss', 'Css', 'Iss', 'Lss', 'wss', 'rss', 'revenue_ss']
table9 = []
for i, label in enumerate(results_labels):
    table9.append([label])
    for y in range(10):
        table9[i].append(((tpi_reform['_DEP_Age'][var_names[i]][y] -
                           tpi_base['_DEP_Age'][var_names[i]][y])
                          / tpi_base['_DEP_Age'][var_names[i]][y]))
    table9[i].extend((
        ((tpi_reform['_DEP_Age'][var_names[i]][:11].sum() -
          tpi_base['_DEP_Age'][var_names[i]][:11].sum())
         / tpi_base['_DEP_Age'][var_names[i]][:11].sum()),
        ((ss_reform['_DEP_Age'][ss_var_names[i]] -
          ss_base['_DEP_Age'][ss_var_names[i]])
         / ss_base['_DEP_Age'][ss_var_names[i]])))
# save table of info to Excel
worksheet = workbook.add_worksheet('Table 9')
worksheet.write(0, 0, 'Macroeconomic Variables')
for y in range(2018, 2028):
    worksheet.write(0, y - 2017, str(y))
worksheet.write(0, 11, '2018-2027')
worksheet.write(0, 12, 'SS')
for i, val in enumerate(table9):
    for j, val2 in enumerate(table9[i]):
        worksheet.write(i + 1, j, val2)

workbook.close()

'''
FIGURES
'''

"""
This version: 12 May 2018
Written by Kerk Phillips

This program fits tax functions for "Integrating Microsimulation Models of Tax
Policy into a DGE Macroeconomic Model: A Canonical Example," by DeBaker, Evans
and Phillips.

This file fits data from Tax Calc to three different functions, compares the
goodness-of-fit, and plots the functions against the data.  The data sample
can be restricted by the age and the capital income of the individuals.
"""



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
    print('GS SSE: ', wsse)
    print('coeffs = ', phi0, phi1, phi2)
    return wsse


'''
The functions below call those above and return the square-root of the sum of
squared errors (SSE) for each model
'''
# load data from pkl file then clean it up (like in txfunc.py)
micro_data = pickle.load(open("micro_data_2017Law.pkl", "rb"),
                         encoding='latin1')
data_orig = micro_data['2018']
data_orig['Total Labor Income'] = \
    (data_orig['Wage income'] +
     data_orig['SE income'])
data_orig['Effective Tax Rate'] = \
    (data_orig['Total tax liability'] /
     data_orig["Adjusted total income"])
data_orig["Total Capital Income"] = \
    (data_orig['Adjusted total income'] -
     data_orig['Total Labor Income'])
# use weighted avg for MTR labor - abs value because
# SE income may be negative
data_orig['MTR Labor'] = \
    (data_orig['MTR wage income'] * (data_orig['Wage income'] /
     (data_orig['Wage income'].abs() +
     data_orig['SE income'].abs())) +
     data_orig['MTR SE income'] *
     (data_orig['SE income'].abs() /
     (data_orig['Wage income'].abs() +
      data_orig['SE income'].abs())))
data = data_orig[['Age', 'MTR Labor', 'MTR capital income',
                  'Total Labor Income', 'Total Capital Income',
                  'Adjusted total income', 'Effective Tax Rate',
                  'Weights']]
# Clean up the data by dropping outliers
# drop all obs with ETR > 0.65
data_trnc = \
    data.drop(data[data['Effective Tax Rate'] > 0.65].index)
# drop all obs with ETR < -0.15
data_trnc = \
    data_trnc.drop(data_trnc[data_trnc['Effective Tax Rate']
                             < -0.15].index)
# drop all obs with ATI, TLI, TCI < $5
data_trnc = data_trnc[(data_trnc['Adjusted total income'] >= 5)
                      & (data_trnc['Total Labor Income'] >= 5) &
                      (data_trnc['Total Capital Income'] >= 5)]

# drop all obs with MTR on capital income > 10.99
data_trnc = \
    data_trnc.drop(data_trnc[data_trnc['MTR capital income']
                             > 0.99].index)
# drop all obs with MTR on capital income < -0.45
data_trnc = \
    data_trnc.drop(data_trnc[data_trnc['MTR capital income']
                             < -0.45].index)
# drop all obs with MTR on labor income > 10.99
data_trnc = data_trnc.drop(data_trnc[data_trnc['MTR Labor']
                                     > 0.99].index)
# drop all obs with MTR on labor income < -0.45
data_trnc = data_trnc.drop(data_trnc[data_trnc['MTR Labor']
                                     < -0.45].index)
data_baseline = data_trnc

# calculate total taxes
data_baseline['Taxes'] = (data_baseline['Effective Tax Rate'] *
                          data_baseline['Adjusted total income'])

# get time-series of interest  - for 43 year old in 2017
data_to_use = data_baseline[data_baseline['Age'] == 42].copy()
y = data_to_use['Total Capital Income'].values
x = data_to_use['Total Labor Income'].values
I = data_to_use['Adjusted total income'].values
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
npts = len(I)  # points in grid
maxinc = 300000.  # upper end of grid
mininc = 0.  # lower end of grid
ygrid = np.zeros(npts)  # this is just a place holder to pass
xgrid = np.linspace(mininc, maxinc, npts)
datagrid = (ygrid, xgrid)  # tuple to pass to functions
Gcoeffs = np.zeros((3,))
Gcoeffs[0] = (tax_func_params_base['_GS_Age']['tfunc_etr_params_S'][21, 0, 0])
Gcoeffs[1] = (tax_func_params_base['_GS_Age']['tfunc_etr_params_S'][21, 0, 1])
Gcoeffs[2] = (tax_func_params_base['_GS_Age']['tfunc_etr_params_S'][21, 0, 2])
# put in terms of ETR by dividing by xgrid
Gyfit = ((Gcoeffs[0] * (xgrid - (xgrid ** (-Gcoeffs[1]) + Gcoeffs[2]) **
                        (-1 / Gcoeffs[1]))) / xgrid)

A = (tax_func_params_base['_DEP_Age']['tfunc_etr_params_S'][21, 0, 0])
B = (tax_func_params_base['_DEP_Age']['tfunc_etr_params_S'][21, 0, 1])
C = (tax_func_params_base['_DEP_Age']['tfunc_etr_params_S'][21, 0, 2])
D = (tax_func_params_base['_DEP_Age']['tfunc_etr_params_S'][21, 0, 3])
max_x = (tax_func_params_base['_DEP_Age']['tfunc_etr_params_S'][21, 0, 4])
min_x = (tax_func_params_base['_DEP_Age']['tfunc_etr_params_S'][21, 0, 5])
max_y = (tax_func_params_base['_DEP_Age']['tfunc_etr_params_S'][21, 0, 6])
min_y = (tax_func_params_base['_DEP_Age']['tfunc_etr_params_S'][21, 0, 7])
shift_x = (tax_func_params_base['_DEP_Age']['tfunc_etr_params_S'][21, 0, 8])
shift_y = (tax_func_params_base['_DEP_Age']['tfunc_etr_params_S'][21, 0, 9])
shift = (tax_func_params_base['_DEP_Age']['tfunc_etr_params_S'][21, 0, 10])
share = (tax_func_params_base['_DEP_Age']['tfunc_etr_params_S'][21, 0, 11])
print('Main spec params = ',
      tax_func_params_base['_DEP_Age']['tfunc_etr_params_S'][21, 0, :])

X = xgrid.copy()
Y = xgrid.copy()*0.0
X2 = X ** 2
Y2 = Y ** 2
tau_x = (((max_x - min_x) * (A * X2 + B * X) / (A * X2 + B * X + 1))
         + min_x)
tau_y = (((max_y - min_y) * (C * Y2 + D * Y) / (C * Y2 + D * Y + 1))
         + min_y)
DEP_fit_nocapital = ((((tau_x + shift_x) ** share) * ((tau_y + shift_y)
                                                      ** (1 - share))) +
                     shift)

X = xgrid.copy() * 0.7
Y = xgrid.copy() * 0.3
X2 = X ** 2
Y2 = Y ** 2
tau_x = (((max_x - min_x) * (A * X2 + B * X) / (A * X2 + B * X + 1))
         + min_x)
tau_y = (((max_y - min_y) * (C * Y2 + D * Y) / (C * Y2 + D * Y + 1))
         + min_y)
DEP_fit_nolabor = (((tau_x + shift_x) ** share) * ((tau_y + shift_y) **
                                                   (1 - share))) + shift

# plot data
# plt.plot(I, etr_data, '.b', ms = 1, label='data')
plt.plot(I, etr_data, '.b', label='data')
plt.plot(xgrid, Gyfit, 'm-', lw=1, label='GS')
plt.plot(xgrid, DEP_fit_nocapital, '-', color='orange', lw=1,
         label='DEP, no y')
plt.plot(xgrid, DEP_fit_nolabor, '-', color='green', lw=1,
         label='DEP, mixed')
plt.xlabel('Total Income')
plt.ylabel('Effective Tax Rate')
# set axes range
plt.xlim(0, 100000)
plt.ylim(-0.2, 0.32)
plt.legend(loc='lower right')
plt.suptitle('Age = 43, Year = 2018')
# save high quality version to external file
plt.savefig('Compare_ETR_functions.png')
plt.show()

plt.plot(I, etr_data, '.b', label='data')
plt.plot(xgrid, DEP_fit_nocapital, '-', color='orange', lw=1,
         label='DEP, no y')
plt.plot(xgrid, DEP_fit_nolabor, '-', color='green', lw=1,
         label='DEP, mixed')
plt.xlabel('Total Income')
plt.ylabel('Effective Tax Rate')
# set axes range
plt.xlim(0, 100000)
plt.ylim(-0.2, 0.32)
plt.legend(loc='lower right')
plt.suptitle('Age = 43, Year = 2018')
# save high quality version to external file
plt.savefig('Compare_DEP_ETR_functions.png')
plt.show()
