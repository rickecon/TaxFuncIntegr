import json
import os
import collections as collect
import six
import re
import pickle

# import ogusa
from ogusa import SS, TPI



class OGUSA(spec1, spec2):
    '''
    OGUSA class.

    Args:
    spec1 = parameterization of baseline
    spec2 = parameterization of reform

    '''
    def __init__(self):
        self.spec1 = spec1
        self.spec2 = spec2

    def runner(self):
        '''
        Method to run the model
        '''
        # baseline compute
        self.base_ss_outputs = SS.run_SS(self.spec1)
        if self.spec1.time_path:
            self.base_tpi_output = TPI.run_TPI(self.spec1)
        # reform compute
        # a bit more complicated because will need stuff from baseline

        self.reform_ss_outputs = SS.run_SS(self.spec2)
        if self.spec2.time_path:
            self.reform_tpi_output = TPI.run_TPI(self.spec2)

    def graphing():
        '''
        Graphing methods
        '''
    # Attributes of OGUSA class:
    # - ss outputs (base and reform)
    # - tpi outputs (base and reform)
    # - spec1 and spec2 parameterizations
    # - taxcalc and ogusa versions
