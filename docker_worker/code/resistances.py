import math  # type: ignore
import numpy as np
from sorcery import dict_of  # type: ignore
import sys

INFINITE = sys.float_info.max


def subplus(arr):
    'negative numbers in array are set to zero'
    arr[arr < 0] = 0
    return arr


class Resistances:
    def __init__(self, canopy3, atm3, id_toy):
        self.atm3 = atm3
        self.canopy3 = canopy3
        self.id_toy = id_toy
        super().__init__()

    def resistances(self):
        """
        Calculates various resistances related to atmospheric conditions and
        canopy structure.

        This function calculates the following resistances:
        - atmospheric_resistance (s/m)
        - in_canopy_atm_resistance (s/m)
        - boundary_layer_res_O3 (s/m)
        - boundary_layer_res_CO2 (s/m)
        - boundary_layer_res_H2O (s/m)
        - cuticolar_resistance_O3 (s/m)
        - cuticolar_resistance_SO2 (s/m)
        - US=Ustar modelled (s/m)
        - Monin_Obukhov_Length (m); atmosphere stability
        - aereodynamic conductance (m/s)

        The function takes into account various input parameters such as:
        - solar radiation (W m-2)
        - wind speed at top of the canopy (m/s)
        - relative humidity at canopy top ()
        - air temperature at canopy top (K)
        - air pressure (pa)
        - friction velocity (m / s)
        - canopy layers
        - leaf area index
        - leaf area index fraction per layer
        - distance from canopy top
        """

        atm3 = self.atm3
        canopy3 = self.canopy3
        id_toy = self.id_toy

        displac = (
            canopy3["disp_height"] * canopy3["height"]
        )  # %zero  displacement height (m)

        # roughness length (m)
        # z_0 = canopy3["rough_lenght"] * canopy3["height"]

        # roughness length (m)
        # WRF model default (urban parks) Shen et al. 2023
        z_0 = 0.5
        t_airk0 = atm3["Tairk0"][id_toy]
        ws0 = atm3["ws0"][id_toy]
        ws0 = max(ws0, 1e-12)
        rh0 = atm3["RH0"][id_toy]
        p_pa = atm3["P_Pa"][id_toy]
        p_kpa = atm3["P_kPa"][id_toy]
        solar_rad = atm3["SW_rad"][id_toy]

        ustar = atm3["USTAR"][id_toy]

        t_airc = atm3["Tair_C"][id_toy]

        von_karman_constant = 0.41  # %von Karman Costant

        # ------------------------------------------------------------
        # Calculation of the inverse of L parameter using the Pasquill

        # Golder D. 1972 (standard)
        # Relations among Stability Parameters in the Surface Layer.
        # Boundary-Layer Meteorology 3 (1): 47–58.

        # It's implemented and Approximation of Golder
        # Zannetti, Paolo. 2013. Air Pollution Modeling:
        # Theories, Computational Methods and Available Software.
        # Springer Science & Business Media.
        st_a = -0.0875 * (z_0**-0.1029)  # Extremely unstable
        st_b = -0.03849 * (z_0**-0.1714)  # unstable
        st_c = -0.0807 * (z_0**-0.3049)  # slighty unstable
        st_d = 0 * (z_0**-0)  # neutral
        st_e = 0.0807 * (z_0**-0.3049)  # Stable
        st_f = 0.03849 * (z_0**-0.1714)  # Very Stable

        schm_o3 = 1.08  # %Launianen et al. 2013.
        schm_co2 = 0.92  # %Launianen et al. 2013.
        prandl = 0.72  # %Prandl number

        schm_no2 = 0.98  # % Novak oppure 1.07
        schm_so2 = 1.15  # % Novak oppure 1.25
        schm_co = 0.76  # % Novak

        # USTAR LOOP
        if math.isnan(ustar):
            # Turner, D. Bruce. 1994.
            # Workbook of Atmospheric Dispersion Estimates:
            # An Introduction to Dispersion Modeling. CRC press.

            # Mohan, M. and Siddiqui, T. A., 1998.
            # Analysis of various schemes for the estimation of
            # atmospheric stability classification.
            # Atmospheric Environment, 32 (21), 3775–3781.

            st_ab = np.mean([st_a, st_b])
            st_bc = np.mean([st_b, st_c])
            st_cd = np.mean([st_c, st_d])

            # % 4th column is modified to implement night time in
            # % clear sky condition
            stability_table = np.array(
                [
                    [st_a, st_ab, st_b, st_f],
                    [st_ab, st_b, st_c, st_f],
                    [st_b, st_bc, st_c, st_e],
                    [st_c, st_cd, st_d, st_d],
                    [st_c, st_d, st_d, st_d],
                ],
                dtype=object,
            )

        # define indexes to compute Monin Obukhov Length
        if ws0 <= 2:
            ws_index = 0
        elif ws0 <= 3:
            ws_index = 1
        elif ws0 <= 5:
            ws_index = 2
        elif ws0 <= 6:
            ws_index = 3
        elif ws0 > 6:
            ws_index = 4

        # solar radiation index (ranges are empirically defined)
        if solar_rad >= 600:
            solar_index = 0
        elif solar_rad >= 300:
            solar_index = 1
        elif solar_rad >= 20:  # % empirical
            solar_index = 2
        elif solar_rad >= 0:
            solar_index = 3

        # inverse of Monin_Obukhov_Length
        l_inverse = stability_table[ws_index][solar_index]

        atmosphere_stability = 1

        if l_inverse != 0:
            if (1 / l_inverse) < 0:
                atmosphere_stability = 1
            elif (1 / l_inverse) > 0:
                atmosphere_stability = 2
        else:
            atmosphere_stability = 3

        if l_inverse != 0:
            monin_obukhov_length = 1 / l_inverse
        else:
            monin_obukhov_length = INFINITE

        # % --------------------------------------------------------------------
        # Spalart, P. R. and Allmaras, S. R., 1992
        # A one-equation turbulence model for aerodynamic flows.

        # switch atmosphere_stability
        if atmosphere_stability == 1:  # % unstable
            # eq. 9
            x_m = (1 - 16 * canopy3["height"] / monin_obukhov_length)**0.25
            psi_m = (
                math.log((1 + x_m**2) / 2)
                + 2 * math.log(((1 + x_m) / 2) ** 2)
                - 2 * math.atan(x_m)
                + math.pi / 2
            )
            ustar = (
                ws0 * von_karman_constant / (math.log(
                    canopy3["height"] / z_0
                ) - psi_m)
            )

        elif atmosphere_stability == 2:  # % instable
            psi_m = -5 * canopy3["height"] / monin_obukhov_length
            ustar = (
                ws0 * von_karman_constant / (math.log(
                    canopy3["height"] / z_0
                ) - psi_m)
            )

        elif atmosphere_stability == 3:  # % neutral
            ustar = ws0 * von_karman_constant / math.log(
                canopy3["height"] / z_0
            )

        else:
            monin_obukhov_length = np.NaN

        # convert relative humidity to Pascal
        humidair_pa0 = self.convert_rh2pascal(rh0, t_airk0 - 273.15, p_pa)

        warmhumiditychange = canopy3["warhumiditychange"]
        coolhumiditychange = canopy3["coolhumiditychange"]

        # humidity profile [Pa m-1]
        if t_airk0 > 288:
            deltah = (
                warmhumiditychange / canopy3["height"]
            )
        elif t_airk0 > 278:
            deltah = (
                warmhumiditychange
                - ((288 - t_airk0) / 10)
                * (warmhumiditychange - coolhumiditychange)
            ) / canopy3["height"]
        else:
            deltah = (
                coolhumiditychange / canopy3["height"]
            )

        # Ustar gradient scaled to WS gradient
        layers = int(canopy3["layers"])

        l_depth = np.zeros(layers, dtype=float)
        humidair_pa = np.zeros(layers, dtype=float)
        rh = np.zeros(layers, dtype=float)
        layer_height = np.zeros(layers, dtype=float)
        ws = np.zeros(layers, dtype=float)
        us = np.zeros(layers, dtype=float)

        layer_height = canopy3.layer_height

        # Schaubroeck et al. 2014 from Yi 2008
        # compute wind speed for each layer
        height = canopy3["height"]
        ws = ws0 * np.exp(
            -0.5 * canopy3.lai_layer * (1 - layer_height / height)
        )

        for i in range(layers):
            l_depth[i] = height - layer_height[i]
            humidair_pa[i] = humidair_pa0 + (deltah * l_depth[i])
            rh[i] = rh0  # % no scaling for humidity
            if rh[i] > 100:
                rh[i] = 99

            # TODO: check with workgroup
            # avoid division by zero
            # ws_ratio = 0
            # if ws0 != 0:
            # us_i = ustar * ws_ratio
            # us_i = max(ustar, 1e-12)
            # us[i] = us_i

            ws_ratio = ws[i] / ws0
            us[i] = ustar * ws_ratio



        # -- resistances computation -------------------------------------------
        # CANOPY SHAPE TO MATRIX
        rho_2 = (
            (p_kpa * 100) / 1013 * 28.95 / 0.0821 / (t_airk0)
        )  # % air density  [kg/m3]

        # canopy_lai  = CANOPY.lai
        canopy_lai = canopy3.lai_layer
        res_cutd0_03 = canopy3["res_cutd0_03"]
        res_cutd0_SO2 = canopy3["res_cutd0_so2"]
        res_ac0 = canopy3["res_ac0"]

        atmospheric_resistance = np.zeros(layers, dtype=float)
        boundary_layer_res_O3 = np.zeros(layers, dtype=float)
        boundary_layer_res_CO2 = np.zeros(layers, dtype=float)
        boundary_layer_res_H2O = np.zeros(layers, dtype=float)

        boundary_layer_res_CO = np.zeros(layers, dtype=float)

        boundary_layer_res_SO2 = np.zeros(layers, dtype=float)
        boundary_layer_res_NO2 = np.zeros(layers, dtype=float)
        cuticolar_resistance_O3 = np.zeros(layers, dtype=float)
        cuticolar_resistance_SO2 = np.zeros(layers, dtype=float)
        cuticolar_resistance_NO2 = np.zeros(layers, dtype=float)
        va = np.zeros(layers, dtype=float)

        x = range(layers)

        for i in x:
            # Nowak et al. 1998.
            # “Modeling the Effects of Urban Vegetation on Air Pollution.”
            # In Air Pollution Modeling and Its Application XII, 399–407.
            # Springer.

            # Killus et al. 1984.
            # “Continued Research in Mesoscale Air Pollution Simulation Modeling
            # Volume 5. Refinements in Numerical Analysis, Transport,
            # Chemistry, and Pollutant Removal.” 1984.
            atmospheric_resistance[i] = ws[i] / (ustar**2)

            # Hicks, B. et al. 1987.
            # “A Preliminary Multiple Resistance Routine for Deriving Dry
            # Deposition Velocities from Measured Quantities.”
            # Water, Air, and Soil Pollution 36 (3–4): 311–30.
            # https://doi.org/10.1007/BF00229675.
            boundary_layer_res_O3[i] = (
                (2 / (von_karman_constant * us[i]))
                * (schm_o3 / prandl) ** (2 / 3)
                / (von_karman_constant * us[i])
            )

            boundary_layer_res_CO2[i] = (
                (2 / (von_karman_constant * us[i]))
                * (schm_co2 / prandl) ** (2 / 3)
                / (von_karman_constant * us[i])
            )

            boundary_layer_res_CO[i] = (
                (2 / (von_karman_constant * us[i]))
                * (schm_co / prandl) ** (2 / 3)
                / (von_karman_constant * us[i])
            )

            boundary_layer_res_SO2[i] = (
                (2 / (von_karman_constant * us[i]))
                * (schm_so2 / prandl) ** (2 / 3)
                / (von_karman_constant * us[i])
            )
            boundary_layer_res_NO2[i] = (
                (2 / (von_karman_constant * us[i]))
                * (schm_no2 / prandl) ** (2 / 3)
                / (von_karman_constant * us[i])
            )

            # Calculation of quasi-laminar boundary layer resistanceid_toy
            va[i] = (
                1.983 * 10**(-5)
            ) / rho_2  # kinematic viscosity of air m^2s^-1
            Re = (z_0 * us[i]) / va[i]

            Dw = (
                2.775 * 10**(-6)
                + 4.479 * (10**(-8)) * (t_airk0)
                + 1.656 * (10**(-10)) * (t_airk0)**2
            )  # molecolar diffusivity of water

            Sc = va[i] / Dw  # Schmidt number

            # Schallhart, Simon et al. 2016.
            # “Characterization of Total Ecosystem-Scale Biogenic VOC Exchange
            # at a Mediterranean Oak–Hornbeam Forest.”
            # Atmospheric Chemistry and Physics 16 (11): 7171–94.
            # https://doi.org/10.5194/acp-16-7171-2016.

            # --> Owen, P. R., and W. R. Thomson. 1963.
            # “Heat Transfer across Rough Surfaces.”
            # Journal of Fluid Mechanics 15 (03): 321.
            # https://doi.org/10.1017/S0022112063000288.
            # --> Garland, J. A. 1977.
            # The Dry Deposition of Sulphur Dioxide to Land and Water Surfaces.
            # Proceedings of the Royal Society A: Mathematical, Physical
            # and Engineering Sciences 354 (1678): 245–68.
            # https://doi.org/10.1098/rspa.1977.0066.

            Re_app = Re**0.24
            Sc_app = Sc**0.8
            st_b = 1 / (1.45 * Re_app * Sc_app)
            boundary_layer_res_H2O[i] = (st_b * us[i]) ** (-1)

            # Zhang, L., Brook, J. R., and Vet, R., 2003.
            # A revised parameterization for gaseous dry deposition
            # in air-quality models.
            # Atmospheric Chemistry and Physics, 3 (6), 2067–2082.

            # --> Zhang, Leiming, Jeffrey R. Brook, and Robert Vet. 2002.
            # “On Ozone Dry Deposition—with Emphasis on Non-Stomatal Uptake
            # and Wet Canopies.” Atmospheric Environment 36 (30): 4787–99.

            cuticolar_resistance_O3[i] = res_cutd0_03 / (
                math.exp(0.03 * rh[i]) * (canopy_lai[i] ** (1 / 4)) * us[i]
            )
            cuticolar_resistance_SO2[i] = res_cutd0_SO2 / (
                math.exp(0.03 * rh[i]) * (canopy_lai[i] ** (1 / 4)) * us[i]
            )
            cuticolar_resistance_NO2[i] = 20000 / (
                math.exp(0.03 * rh[i]) * (canopy_lai[i] ** (1 / 4)) * us[i]
            )

            # % @@0 [improvement] Zhang proposes a solution
            # also for wet deposition

        ustar = np.repeat(ustar, len(ws))


        #aereodynamic_conductance = (
        #    (ws * von_karman_constant**2)
        #    / (
        #        (np.log(layer_height / z0_m) - psi_m)
        #        * (np.log(layer_height / z0_h) - psi_h)
        #    )
        #)



        #aereodynamic_conductance = (ustar / ws)**2 * ws

        # Holwerda et al. 2011 from Van der Tol et al. 2003
        # Wet canopy evaporation from a Puerto Rican lower montane rain forest:
        # The importance of realistically estimated aerodynamic conductance.
        # Journal of Hydrology, 414, 1–15.
        # aerodynamic conductance for momentum (for heat)

        # TODO: check with workgroup
        aereodynamic_conductance = ustar**2 / ws

        # TODO: TEST

        # res_ac0 from Zhang, L., Brook, J. R., and Vet, R., 2003.
        # A revised parameterization for gaseous dry deposition in air-quality
        # models. Atmospheric Chemistry and Physics, 3 (6), 2067–2082.
        ustar[ustar == 0] = 1e-12
        in_canopy_atm_resistance = (
            res_ac0 * canopy_lai ** (1 / 4)
        ) / (ustar**2)


        # GET ONLY POTIVE VALUES; OTHERS ARE SET TO 0
        atmospheric_resistance   = subplus(atmospheric_resistance)
        in_canopy_atm_resistance = subplus(in_canopy_atm_resistance)
        boundary_layer_res_O3    = subplus(boundary_layer_res_O3)
        boundary_layer_res_H2O   = subplus(boundary_layer_res_H2O)
        boundary_layer_res_CO2   = subplus(boundary_layer_res_CO2)
        cuticolar_resistance_O3  = subplus(cuticolar_resistance_O3)
        cuticolar_resistance_SO2 = subplus(cuticolar_resistance_SO2)
        boundary_layer_res_CO    = subplus(boundary_layer_res_CO)
        boundary_layer_res_NO2   = subplus(boundary_layer_res_NO2)
        boundary_layer_res_SO2   = subplus(boundary_layer_res_SO2)
        cuticolar_resistance_NO2 = subplus(cuticolar_resistance_NO2)

        ustar = subplus(us)
        ws = subplus(ws)
        humidair_pa = subplus(humidair_pa)
        rh = subplus(rh)
        air_kinematic_viscosity = va

        # Liu, Jiakai, et al. "Dry deposition of particulate matter at an urban
        # forest, wetland and lake surface in Beijing."
        # Atmospheric Environment 125 (2016): 178-187.
        # canopy_height = canopy3["height"]
        # app2 = math.log((canopy_height - displac) / z_0)
        # app2 = np.repeat(app2, len(us))
        # von_karman_constant = np.repeat(von_karman_constant, len(us))

        # mean_wind_velocity = np.divide(us, von_karman_constant)
        # mean_wind_velocity = np.multiply(mean_wind_velocity, app2)
        mean_wind_velocity = ws

        [z_bl, rho_boundary] = self.compute_boundary_layer(p_pa, t_airc, rh)

        res = dict_of(
            atmospheric_resistance,
            in_canopy_atm_resistance,
            boundary_layer_res_O3,
            boundary_layer_res_H2O,
            boundary_layer_res_CO2,
            boundary_layer_res_NO2,
            boundary_layer_res_SO2,
            boundary_layer_res_CO,
            layer_height,
            cuticolar_resistance_O3,
            cuticolar_resistance_NO2,
            cuticolar_resistance_SO2,
            ustar,
            ws,
            humidair_pa,
            air_kinematic_viscosity,
            mean_wind_velocity,
            rh,
            monin_obukhov_length,
            deltah,
            l_depth,
            rho_2,
            aereodynamic_conductance,
            z_bl,
            rho_boundary,
        )
        return res

    def convert_rh2pascal(
            self,
            relative_humidity: float,
            celsius: float,
            p_pa: float
    ) -> float:
        """
        Convert relative humidity to Pascal.

        Parameters:
        relative_humidity (float): The relative humidity in percentage.
        celsius (float): The temperature in Celsius.
        p_pa (float): The pressure in Pascal.

        Returns:
        float: The converted relative humidity in Pascal.

        Raises:
        None.

        Example:
        >>> convert_rh2pascal(50, 25, 101325)
        1266.6666666666667
        """

        gkg = 0.01 * relative_humidity * (
            -0.082 + math.exp(1.279 + 0.0687 * celsius)
        )
        gkg001 = gkg * 0.001
        wat_air_ratio = 18.016 / 28.97  # molar ratio water to air
        rh2 = (gkg001 / (gkg001 + wat_air_ratio)) * p_pa
        return rh2


    def rel_to_abs(self, T,P,RH):
        """Returns absolute humidity given relative humidity.

        Inputs:
        --------
        T : float
            Absolute temperature in units Kelvin (K).
        P : float
            Total pressure in units Pascals (Pa).
        RH : float
            Relative humidity in units percent (%).

        Output:
        --------
        absolute_humidity : float
            Absolute humidity in units [kg water vapor / kg dry air].

        References:
        --------
        1. Sonntag, D. "Advancements in the field of hygrometry". 1994. https://doi.org/10.1127/metz/3/1994/51
        2. Green, D. "Perry's Chemical Engineers' Handbook" (8th Edition). Page "12-4". McGraw-Hill Professional Publishing. 2007.

        Version: 0.0.1
        Author: Steven Baltakatei Sandoval
        License: GPLv3+
        """

        import math

        # Check input types
        #T = float(T)
        #P = float(P)
        #RH = float(RH)

        # Set constants and initial conversions
        epsilon = 0.62198 # (molar mass of water vapor) / (molar mass of dry air)
        t = T - 273.15 # Celsius from Kelvin

        P_hpa = P / 100 # hectoPascals (hPa) from Pascals (Pa)

        # Calculate e_w(T), saturation vapor pressure of water in a pure phase, in Pascals
        ln_e_w = -6096*T**-1 + 21.2409642 - 2.711193*10**-2*T + 1.673952*10**-5*T**2 + 2.433502*np.log(T) # Sonntag-1994 eq 7 e_w in Pascals
        e_w = np.exp(ln_e_w)
        e_w_hpa = e_w / 100 # also save e_w in hectoPascals (hPa)

        # Calculate f_w(P,T), enhancement factor for water
        f_w = 1 + (10**-4*e_w_hpa)/(273 + t)*(((38 + 173*np.exp(-t/43))*(1 - (e_w_hpa / P_hpa))) + ((6.39 + 4.28*np.exp(-t / 107))*((P_hpa / e_w_hpa) - 1))) # Sonntag-1994 eq 22.

        # Calculate e_prime_w(P,T), saturation vapor pressure of water in air-water mixture, in Pascals
        e_prime_w = f_w * e_w # Sonntag-1994 eq 18

        # Calculate e_prime, vapor pressure of water in air, in Pascals
        e_prime = (RH / 100) * e_prime_w

        # Calculate r, the absolute humidity, in [kg water vapor / kg dry air]
        r = (epsilon * e_prime) / (P - e_prime)

        return r

    def compute_boundary_layer(self, p_pa, celsius, relative_humidity):
        celsius = np.repeat(celsius, len(relative_humidity))

        # Absolute hum [kg m-3]
        rho_boundary = p_pa / (287.05 * (273.15 + celsius))

        # lawrence (2005)
        if (50 <= relative_humidity[0] <= 100) and (0 < celsius[0] < 30):
            z_bl = 0 + (20 + celsius / 5) * (100 - relative_humidity)
        else:
            z_bl = 25 * (100 - relative_humidity)
        return z_bl, rho_boundary

