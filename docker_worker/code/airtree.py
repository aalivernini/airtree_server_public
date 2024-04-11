'''
airtree core module, running the model for mobile and wep app
'''

import numpy as np
from modeldata import ModelData
import carbon
from radiation import Radiation
from resistances import Resistances
from energybalance import EnergyBalance
from phenology import Phenology
from fluxes import Fluxes
from base import DotDict, get_totals_web


class Airtree():
    '''
    Airtree core class for simulating fluxes (carbon and pollutants) of a tree

    This class represents the airtree model, which is used to simulate
    the carbon, the energy balance and the fluxes (eg. pollutants removed)
    The model takes input data for the project, canopy and atmospheric
    conditions, and runs the simulation for a given time frequency.

    parameters:
    - project (dict): a dictionary containing project-specific data.
    - canopy (dict): a dictionary containing tree-specific data.
    - atm (dataframe): a dataframe containing atmospheric and pollutants data.
    - frequenza (int, optional): the frequency of the simulation in days.
        default is 6.
    '''

    def __init__(self, project, canopy, atm, frequenza=6):
        self.project      = DotDict(project)
        self.canopy       = DotDict(canopy)
        self.atm          = atm
        self.airtree_mode = self.project.mode
        self.frequenza    = frequenza

    def airtree(self):
        '''
        Run the airtree model.

        Parameters:
        - self: The instance of the class.

        Returns:
        - results: A dictionary containing the results of the model.

        The function starts by converting the input data to the airtree standard
        using the ModelData class.
        It then initializes LAI correction mode for phenology using the Phenology class.

        The model is then processed for each line in the ATM dataset.
        The function iterates over the time steps in the dataset
        and performs the following steps for each time step:

        - Apply phenology correction to the LAI using the  Phenology class.
        - Calculate radiation using the Radiation class.
        - Calculate resistances using the Resistances class.
        - Calculate energy balance using the EnergyBalance class.
        - Calculate carbon sink

        Finally the model computes the fluxes using the Fluxes class
        '''

        frequenza = self.frequenza
        canopy3    = self.canopy
        atm3       = self.atm
        project3   = self.project

        # CONVERT INPUT DATA TO AIRTREE STANDARD FORMAT
        model = ModelData(project3, canopy3, atm3, 1)
        [canopy3, atm3] = model.modeldata()

        # INIT LAI CORRECTION MODE FOR PHENOLOGY
        is_deciduous = not canopy3['evergreen']

        pheno = Phenology(
            is_deciduous,
            self.project['lat'],
            self.project['lon'],
            canopy3['leafing'],
            canopy3['defoliation'],
            atm3,
        )

        # CREATE RESULT DICTIONARY
        result3 = DotDict()
        key2 = ['rad3', 'res3', 'eb3', 'carbon3', 'tree3']
        for key in key2:
            result3[key] = DotDict()
        result3.tree3.lai = []

        # number of loops for each day
        daytimes = 48 / frequenza

        # PROCESSING MODEL FOR EACH LINE IN ATM DATASET
        time_steps = atm3["TOY"].len()
        ixx = 0
        while ixx < time_steps:

            # PHENOLOGY
            canopy3["lai"] = pheno.apply_correction(canopy3["max_lai"], ixx)
            canopy3.lai_layer = canopy3.lai * canopy3['layer_volume_ratio']

            # RADIATION
            radiation1 = Radiation(
                project3,
                canopy3,
                atm3,
                ixx
            )
            rad1 = radiation1.radiation()

            # RESISTANCES
            resistance1 = Resistances(
                canopy3,
                atm3,
                ixx
            )
            res1 = resistance1.resistances()

            # ENERGY BALANCE
            energybalance1 = EnergyBalance(
                project3,
                canopy3,
                atm3,
                rad1,
                res1,
                ixx
            )
            ceb1 = energybalance1.energybalance()

            # CARBON SINK
            carbon1 = carbon.CarbonBalance(canopy3, ceb1, daytimes)
            carb1 = carbon1.carbon_balance()

            # POPULATE RESULT DICTIONARIES
            result1 = {
                'rad3'    : rad1,
                'res3'    : res1,
                'eb3'     : ceb1,
                'carbon3' : carb1,
            }
            if ixx == 0:
                for key_l1 in result1:
                    for key_l2 in result1[key_l1]:
                        result3[key_l1][key_l2] = []

            for key_l1 in result1:
                for key_l2 in result1[key_l1]:
                    result3[key_l1][key_l2].append(result1[key_l1][key_l2])
            result3.tree3.lai.append(canopy3.lai)

            # NEXT TIME STEP
            ixx = ixx + 1

        # CONVERT DATA TO NUMPY ARRAY
        for key_l1 in result3:
            for key_l2 in result3[key_l1]:
                result3[key_l1][key_l2] = np.array(
                    result3[key_l1][key_l2]
                ).squeeze()


        # FLUXES
        model_fluxes = Fluxes(
            atm3,
            canopy3,
            result3.rad3,
            result3.res3,
            result3.eb3,
            result3.tree3,
            frequenza
        )
        fluxes_voc3, fluxes_pollut = model_fluxes.fluxes()

        # STORE RESULTS
        results = {}
        mode = self.airtree_mode

        if mode == 'web':
            results = get_totals_web(
                fluxes_pollut,
                result3.carbon3,
                atm3,
                frequenza
            )
        elif mode == 'debug':
            results = {
                'ATM'           : atm3,
                'CANOPY'        : canopy3,
                'RAD'           : result3.rad3,
                'RES'           : result3.res3,
                'EB'            : result3.eb3,
                'CARBON'        : result3.carbon3,
                'FLUXES_VOC'    : fluxes_voc3,
                'FLUXES_POLLUT' : fluxes_pollut,
                'TREE'          : result3.tree3,
            }
        else:
            raise NotImplementedError(f'mode not implemented: {mode}')
        return results
