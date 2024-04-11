import numpy as np
import math
# from pvlib import tools
# from datetime import datetime
# from pvlib import location
# from pvlib.tools import sind
from sorcery import dict_of
from typing import Tuple


class Radiation():
    def __init__(self, project3, canopy3, atm3, id_toy):
        self.atm3 = atm3
        self.canopy3 = canopy3
        self.project3 = project3
        self.id_toy = id_toy
        super().__init__()

    def radiation(self):
        """
        Calculates the radiation components (PAR, NIR, UV)
        for a given location and time.
        """
        project3 = self.project3
        canopy3 = self.canopy3
        atm3 = self.atm3
        id_toy = self.id_toy

        toy1 = atm3["TOY"][id_toy]
        # dtime = datetime.utcfromtimestamp(atm3["TIMEs"][id_toy])

        # loc = location.Location(
        #     latitude=project3["lat"],
        #     longitude=project3["lon"],
        #     altitude=project3["altitude"],
        # )
        # spos = loc.get_solarposition(dtime)
        # sun_elevation = spos['elevation'].iloc[0]

        # sun_elevation = max(sun_elevation, 0)
        # sun_zenith = 90 - sun_elevation

        # sinbeta = sind(sun_elevation)
        # sinbeta = max(sinbeta, 0.0001)
        sun_elevation = atm3["sun_elevation"][id_toy]
        sun_zenith = atm3["sun_zenith"][id_toy]
        sinbeta = atm3["sinbeta"][id_toy]

        radiazione = atm3["SW_rad"]

        [clearness_index,
         diffuse_radiation_fraction] = self.get_diffuse_fraction(
            id_toy,
            toy1,
            sun_zenith,
            radiazione
        )

        [uv, par, nir] = self.get_radiation_components(
            clearness_index,
            radiazione[id_toy]
        )

        lai_matrix = np.repeat(canopy3["lai"], len(canopy3['layer_volume_ratio']))
        shape_matrix = canopy3['layer_volume_ratio']
        cumulative_lai  = lai_matrix * shape_matrix
        for index in range(1, len(cumulative_lai)):
            cumulative_lai[index] = (
                cumulative_lai[index] + cumulative_lai[index - 1]
            )

        canopy_cumulative_depth  = canopy3['height'] - canopy3['layer_height']

        if canopy3.leafangle != -1:
            # G = tools.cosd(canopy3.leafangle)
            G = np.cos(np.deg2rad(canopy3.leafangle))
        else:
        #  for spheric distribution
            G = np.ones(len(sinbeta)) * 0.5;

        # Goudriaan, J. Crop Micrometeorology: A Simulation Study.
        # Simulation Monographs. Wageningen: Pudoc, Center for
        # Agricultural Publishing and Documentation, 1977.
        kb = G *  canopy3["clumping_index"] / sinbeta

        # max of kb taken from Zhang, L. et al. 2014.
        # A meta-analysis of the canopy light extinction coefficient in
        # terrestrial ecosystems. Frontiers of Earth Science, 8 (4), 599–609.
        if canopy3["conifer"]:
            kb_max = 0.45 + 0.11 * 2
            kb_min = 0.45 - 0.11 * 2
        else:
            kb_max = 0.59 + 0.12 * 2
            kb_min = 0.59 - 0.12 * 2

        kb = min(kb, kb_max)
        kb = max(kb, kb_min)

        reflectance_uv = ((1 - (
            1 - canopy3["scattering_uv"] - canopy3["transmittance_uv"]
        ) ** 0.5) / (
            1 + (1 - canopy3["scattering_uv"]) ** 0.5
        )) * 2 / (1 + 1.6 * math.sin(math.radians(sun_elevation)))

        reflectance_vis = ((1 - (
            1 - canopy3["scattering_vis"] - canopy3["transmittance_vis"]
        ) ** 0.5) / (1 + (
            1 - canopy3["scattering_vis"] - canopy3["transmittance_vis"]
        ) ** 0.5)) * 2 / (1 + 1.6 * math.sin(math.radians(sun_elevation)))

        reflectance_nir = ((1 - (
            1 - canopy3["scattering_nir"] - canopy3["transmittance_nir"]
        ) ** 0.5) / (1 + (
            1 - canopy3["scattering_nir"] - canopy3["transmittance_nir"]
        ) ** 0.5)) * 2 / (1 + 1.6 * math.sin(math.radians(sun_elevation)))

        transmittance_vis = canopy3["transmittance_vis"]
        transmittance_nir = canopy3["transmittance_nir"]
        transmittance_uv  = canopy3["transmittance_uv"]

        # Bodin, P., and O. Franklin. 2012. “Efficient Modeling of Sun/shade
        # Canopy Radiation Dynamics Explicitly Accounting for Scattering.”
        # Geoscientific Model Development 5 (2): 535–41.
        # doi:10.5194/gmd-5-535-2012.

        # PAR
        [par_shaded,
         par_sun,
         kd_par_sun,
         fraction_sun_leaves] = self.model_shortwave(
            cumulative_lai,
            reflectance_vis,
            transmittance_vis,
            canopy3.scattering_vis,
            par,
            diffuse_radiation_fraction,
            kb,
            canopy_cumulative_depth,
            canopy3
        )
        if np.any(par_sun < 0):
            breakpoint()


        # NIR
        [nir_shaded,
         nir_sun,
         kd_nir_sun,
         fraction_sun_leaves] = self.model_shortwave(
             cumulative_lai,
             reflectance_nir,
             transmittance_nir,
             canopy3.scattering_nir,
             nir,
             diffuse_radiation_fraction,
             kb,
             canopy_cumulative_depth,
             canopy3
        )

        # UV
        [uv_shaded,
         uv_sun,
         kd_uv_sun,
         fraction_sun_leaves] = self.model_shortwave(
             cumulative_lai,
             reflectance_uv,
             transmittance_uv,
             canopy3.scattering_uv,
             uv,
             diffuse_radiation_fraction,
             kb,
             canopy_cumulative_depth,
             canopy3
        )

        rad_tot_shaded = np.array(uv_shaded + par_shaded + nir_shaded)
        rad_tot_sun = np.array(uv_sun + par_sun + nir_sun)

        rad_tot = np.array(rad_tot_shaded + rad_tot_sun)
        par_tot = np.array(par_sun + par_shaded)
        par_tot_sum = sum(par_tot)
        uv_shaded = np.array(uv_shaded)
        par_shaded = np.array(par_shaded)
        nir_shaded = np.array(nir_shaded)
        uv_sun = np.array(uv_sun)
        nir_sun = np.array(nir_sun)
        par_sun = np.array(par_sun)

        cumulative_absorbed_par = np.zeros(len(par_tot))
        cumulative_absorbed_par[0] = par_tot[0]
        for index_par in range(1, canopy3.layers):
            cumulative_absorbed_par[index_par] = (
                cumulative_absorbed_par[index_par - 1] + par_tot[index_par]
            )

        layer_available_par = np.zeros(len(par_tot))
        layer_available_par[0] = par
        for index_par in range(1, canopy3.layers):
            layer_available_par[index_par] = (
                par - cumulative_absorbed_par[index_par]
            )

        # convert measures in umol m-2 s-1
        layer_available_par_umol = layer_available_par * 4.17
        rad_tot_umol             = rad_tot * 4.17
        par_tot_sum_umol         = par_tot_sum * 4.17
        par_sun_umol             = par_sun * 4.17
        par_shaded_umol          = par_shaded * 4.17

        rad = dict_of(
            rad_tot_shaded,
            rad_tot_sun,
            rad_tot,
            par_tot,
            uv_shaded,
            uv_sun,
            par_shaded,
            par_sun,
            nir_shaded,
            nir_sun,
            sun_elevation,
            uv,
            par,
            nir,
            fraction_sun_leaves,
            clearness_index,
            diffuse_radiation_fraction,
            sun_zenith,
            reflectance_uv,
            reflectance_nir,
            reflectance_vis,
            transmittance_uv,
            transmittance_nir,
            transmittance_vis,
            par_tot_sum,
            layer_available_par,
            layer_available_par_umol,
            rad_tot_umol,
            par_tot_sum_umol,
            par_sun_umol,
            par_shaded_umol,
            kd_uv_sun,
            kd_par_sun,
            kd_nir_sun,
            kb
        )
        return rad

    def model_shortwave(
        self,
        cumulative_LAI,
        reflectance,
        transmittance,
        scattering,
        incoming_radiation,
        diffuse_radiation_fraction,
        kb,
        canopy_cumulative_depth,
        canopy3
    ):
        """
        Calculates the shortwave radiation absorbed by shaded and sunlit leaves
        in a canopy.

        Parameters:
        cumulative_LAI (array-like): The cumulative leaf area index (LAI)
            at each layer of the canopy.
        reflectance (float): The reflectance of the canopy.
        transmittance (float): The transmittance of the canopy.
        scattering (float): The scattering coefficient of the canopy.
        incoming_radiation (float): The total incoming radiation.
        diffuse_radiation_fraction (float): The fraction of incoming radiation
            that is diffuse.
        kb (float): The extinction coefficient of the canopy.
        canopy_cumulative_depth (float): The cumulative depth of the canopy.
        CANOPY (object): An object containing information about the canopy.

        Returns:
        rad_shaded_leaves (array-like): The shortwave radiation absorbed
            by shaded leaves at each layer of the canopy.
        rad_sunlit_leaves (array-like): The shortwave radiation absorbed
            by sunlit leaves at each layer of the canopy.
        kd (float): The diffuse attenuation coefficient.
        fraction_sun_leaves (array-like): The fraction of sunlit leaves
            at each layer of the canopy.

        References:
        Bodin, P. and Franklin, O., 2012. Efficient modeling of sun/shade canopy
        radiation dynamics explicitly accounting for scattering.
        Geoscientific Model Development, 5 (2), 535–541.
        """
        lai_layer = canopy3.lai_layer

        # eq. 2
        # Spitters 1986. Separating the diffuse and direct component of global
        # radiation and its implications for modeling canopy photosynthesis
        # Part II. Calculation of canopy photosynthesis
        kd = 0.8 * (1 - scattering) ** 0.5
        # eq. 1
        diffuse_radiation = ((incoming_radiation * diffuse_radiation_fraction)
                             * (1 - reflectance) * np.exp(-kd * cumulative_LAI))
        # eq. 8
        scattering_radiation_down = (
            incoming_radiation
            * (1 - diffuse_radiation_fraction)
            * transmittance
            * (
                np.exp(-kb * canopy_cumulative_depth)
                - np.exp(-kd * canopy_cumulative_depth)
            )
            / (kd - kb)
        )
        # eq. 9
        scattering_radiation_up = (
            incoming_radiation
            * (1 - diffuse_radiation_fraction)
            * reflectance
            * (
                np.exp(-kb * canopy_cumulative_depth)
                - np.exp(
                    kd
                    * canopy_cumulative_depth
                    - (kb + kd) * canopy3.depth
                )
            )

            / (kb + kd)
        )

        # eq. 3
        fraction_sun_leaves = np.exp(-kb * cumulative_LAI)

        # absorbed radiation
        # eq. 14
        rad_shaded_leaves = (
            (1 - fraction_sun_leaves)
            * lai_layer
            * (
                (kd / (1 - scattering) ** 0.5)
                * diffuse_radiation
                + (kd / (1 - reflectance) ** 0.5)
                * scattering_radiation_up
                + (kd / (1 - transmittance) ** 0.5)
                * scattering_radiation_down)
        )
        # eq. 15
        rad_sunlit_leaves = (
            fraction_sun_leaves
            * lai_layer
            * (
                (kd / (1 - scattering) ** 0.5) * diffuse_radiation
                + (kd / (1 - reflectance) ** 0.5) * scattering_radiation_up
                + (kd / (1 - transmittance) ** 0.5) * scattering_radiation_down
                + kb * incoming_radiation * (1 - diffuse_radiation_fraction)
            )
        )
        if np.any(rad_sunlit_leaves < 0):
            breakpoint()
        return rad_shaded_leaves, rad_sunlit_leaves, kd, fraction_sun_leaves

    def get_diffuse_fraction(self, id_toy, doy, zenith, total_radiation):
        '''
        Calculates the diffuse solar fraction
        '''
        extraterrestrial_flux_density = 1367  # W m-2 mean value

        # J. A. Duffie, W. A. Beckman
        # Solar Energy Thermal Processes, John Wiley and Sons, New York, 1980.
        extraterrestrial_radiation = extraterrestrial_flux_density * (
            1 + 0.0333 * math.cos(math.radians(360 * doy / 365))
            * math.cos(math.radians(zenith))
        ) # W m-2
        sun_elevation = 90 - zenith

        # Dervishi et al. (2011)
        clearness_index = total_radiation[id_toy] / extraterrestrial_radiation

        # Reindl DT, Beckman WA, Duffie JA.
        # Diffuse fraction corrections. Solar Energy 1990;45(1):1–7
        sin_solar = math.sin(math.radians(sun_elevation))

        if clearness_index <= 0.3:
            i = 1
        else:
            i = 0

        dsf = 1.02 - 0.254 * clearness_index + 0.0123 * sin_solar

        diffuse_solar_fraction = np.array([dsf])

        if diffuse_solar_fraction > 1:
            i2 = 1
        else:
            i2 = 0

        if (i * i2) > 0:
            index_costraint = 1
            diffuse_solar_fraction[index_costraint-1] = 1
        else:
            index_costraint = 0

        if (0.3 < clearness_index) and  (clearness_index < 0.78):
            i = 1
            diffuse_solar_fraction[i-1] = 1.4 - 1.749 * clearness_index + 0.177 * sin_solar
        else:
            i = 0

        if (i * (diffuse_solar_fraction > 0.97)) > 0:
            index_costraint = 1
            diffuse_solar_fraction[i-1] = 0.97
        else:
            index_costraint = 0

        if (i * (diffuse_solar_fraction < 0.1))> 0:
            index_costraint = 1
            diffuse_solar_fraction[i-1] = 0.1
        else:
            index_costraint = 0

        if clearness_index >= 0.78:
            i = 1
            diffuse_solar_fraction[i-1] = 0.486 * clearness_index - 0.182 * sin_solar
        else:
            i= 0

        if (i * (diffuse_solar_fraction < 0.1)) > 0:
            index_costraint = 1
            diffuse_solar_fraction[i-1] = 0.1
        else:
            index_costraint = 0
        return clearness_index, diffuse_solar_fraction

    def get_radiation_components(
        self,
        clearness_index: float,
        solar_radiation: float,
        is_gldas: bool = True
    ) -> Tuple[float, float, float]:
        """
        Calculates the components of solar radiation (UV, PAR, NIR)
        based on the clearness index and total solar radiation.

        Parameters:
        - clearness_index (float): The clearness index, a dimensionless value
            between 0 and 1.
        - solar_radiation (float): The total solar radiation in W/m^2.

        Returns:
        - uv (float): The UV component of solar radiation in W/m^2.
        - par (float): The PAR (Photosynthetically Active Radiation)
            component of solar radiation in W/m^2.
        - nir (float): The NIR (Near Infrared) component of solar radiation
            in W/m^2.

        The calculation of the radiation components is based on:
        Escobedo, J. F., Gomes, E. N., Oliveira, A. P., and Soares, J., 2009.
        Modeling hourly and daily fractions of UV, PAR and NIR to global solar
        radiation under various sky conditions at Botucatu, Brazil.
        Applied Energy, 86 (3), 299–309.
        """

        if clearness_index <= 0.35:
            if is_gldas:
                solar_radiation = solar_radiation / (0.510 + 0.410)
            uv = 0.049 * solar_radiation
            par = 0.510 * solar_radiation
            nir = 0.410 * solar_radiation

        if 0.35 < clearness_index <= 0.55:
            if is_gldas:
                solar_radiation = solar_radiation / (0.495 + 0.461)
            uv = 0.045 * solar_radiation
            par = 0.495 * solar_radiation
            nir = 0.461 * solar_radiation

        if 0.55 < clearness_index <= 0.65:
            if is_gldas:
                solar_radiation = solar_radiation / (0.490 + 0.469)
            uv = 0.042 * solar_radiation
            par = 0.490 * solar_radiation
            nir = 0.469 * solar_radiation

        if clearness_index > 0.65:
            if is_gldas:
                solar_radiation = solar_radiation / (0.489 + 0.470)
            uv = 0.041 * solar_radiation
            par = 0.489 * solar_radiation
            nir = 0.470 * solar_radiation
        return uv, par, nir



