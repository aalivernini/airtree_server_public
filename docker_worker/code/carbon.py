"""
Carbon computation for airtree model

the net primary production (npp) is computed as ratio of GPP
instead of implementing the GOTILWA+ model
"""

import numpy as np


class CarbonBalance:
    'class for computing carbon balance'

    def __init__(self, canopy3, ceb1, daytimes):
        self.canopy3 = canopy3
        self.ceb = ceb1
        self.daytimes = daytimes
        self.npp_factor = 0.46  # Collalti and Prentice 2019

    def carbon_balance(self):
        '''compute carbon balance
        '''
        canopy3 = self.canopy3
        carbon3 = {}

        tree_area = (
            (canopy3["crownwidth"] / 2)**2 * np.pi
        )

        # --> TOTALS FOR TREE CANOPY
        # conversion from umol(CO2) m-2 s-1 to g(C) m-2 day-1
        conversion_factor = 12.0 * 10**-6.0 * (60 * 60 * 24)
        gpp_day = self.ceb.tot_A * conversion_factor * tree_area  # g(C) day-1
        gpp_ti = gpp_day * 1 / (self.daytimes)  # g(C) in the time interval
        npp_ti = gpp_ti * self.npp_factor       # g(C) in the time interval

        gpp = gpp_day / 24  # g(C)/hour
        npp = gpp * self.npp_factor  # g(C)/hour
        # <--

        gpp_m2 = gpp / tree_area  # g(C) h-1 m-2
        npp_m2 = gpp_m2 * self.npp_factor

        # STORE RESULTS
        carbon3["npp_ti"] = npp_ti
        carbon3["gpp_ti"] = gpp_ti
        carbon3["gpp_tot"] = gpp
        carbon3["npp_tot"] = npp
        carbon3["gpp_m2"] = gpp_m2
        carbon3["npp_m2"] = npp_m2
        # g(C) m-2 in the time interval
        carbon3["gpp_ti_m2"] = gpp_ti / tree_area
        carbon3["npp_ti_m2"] = npp_ti / tree_area
        return carbon3
