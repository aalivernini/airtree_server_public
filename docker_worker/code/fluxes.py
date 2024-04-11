import numpy as np
from dataclasses import dataclass, field
import copy


class Fluxes:
    def __init__(self, atm3, canopy3, rad3, res3, eb3, tree3, frequency):
        self.atm3 = atm3
        self.canopy3 = canopy3
        self.rad3 = rad3
        self.res3 = res3
        self.eb3 = eb3
        self.lai = tree3.lai
        self.frequency = frequency
        self.lai_layer = (
            np.expand_dims(tree3.lai, 1)
            * np.expand_dims(canopy3["layer_volume_ratio"], 0)
        )

    def get_particulate_input(self, canopy3, atm3, res3, eb3, rad3):
        lai_layer = self.lai_layer

        aereodynamic_resistance = 1 / res3["aereodynamic_conductance"]
        broadleaf = not canopy3['conifer']
        arh = canopy3["rough_lenght"] * canopy3["height"]

        input_pm = InputParticulate(
            rain=atm3["Precip"],
            lai=lai_layer,
            wsp=np.array(atm3["ws0"]),
            hcl=canopy3["layer_height"],
            htop=canopy3["height"],

            mean_wind_velocity=res3["mean_wind_velocity"],
            ustar=res3["ustar"],
            leaf_width=canopy3["leafwidth"],
            leaf_length=canopy3["leaflength"],
            leaf_thickness=canopy3["leafthickness"],
            z_bl=res3["z_bl"],
            aereodynamic_resistance=aereodynamic_resistance,
            aereodynamic_roughness_height=arh,
            air_temperature=atm3["Tairk0"],
            broadleaf=broadleaf,
            fraction_sun_leaves=rad3["fraction_sun_leaves"],
            hsn_lh_evap=eb3["hsn_lh_evap"],
            hsd_lh_evap=eb3["hsd_lh_evap"],
        )
        i_pm10 = copy.deepcopy(input_pm)
        i_pm25 = copy.deepcopy(input_pm)
        i_pm01 = copy.deepcopy(input_pm)

        pm10_conc = np.array(atm3["pm10"])   # [μg m⁻³] PM 10
        pm25_conc = np.array(atm3["pm2_5"])  # [μg m⁻³] PM 2.5
        pm01_conc = np.array(atm3["pm1"])    # [μg m⁻³] PM 1

        # [kg m⁻³] Abhijith and Kumar 2020
        # Traffic related density Cai et al. 2020 from (Hu et al., 2012; Yin et al., 2015)
        particulate_density = 1.5e3 * np.ones(lai_layer.shape)

        # meam diameter for PM10, PM2.5, PM1
        pm10_d = (2.5 + 10) / 2e6
        pm25_d = (1 + 2.5) / 2e6
        pm01_d = (0.1 + 1) / 2e6  # diameters below 0.1 μm are ultrafine particles

        i_pm10.setup(pm10_d, pm10_conc, particulate_density)
        i_pm25.setup(pm25_d, pm25_conc, particulate_density)
        i_pm01.setup(pm01_d, pm01_conc, particulate_density)
        return (i_pm10, i_pm25, i_pm01)

    def get_particulate_results(self, i_pm10, i_pm25, i_pm01):
        # --> compute particulate fluxes
        # PM10
        pm10_model = ParticulateModel(i_pm10)
        water = pm10_model.get_water()  # valid for all particulate dimensions
        pm10_ddv = pm10_model.get_dry_deposition_velocity()
        pm10r = pm10_model.get_pm(water, pm10_ddv)

        # PM2.5
        pm25_model = ParticulateModel(i_pm25)
        water = pm25_model.get_water()  # valid for all particulate dimensions
        pm25_ddv = pm25_model.get_dry_deposition_velocity()
        pm25r = pm25_model.get_pm(water, pm25_ddv)

        # PM1
        pm01_model = ParticulateModel(i_pm01)
        water = pm01_model.get_water()  # valid for all particulate dimensions
        pm01_ddv = pm01_model.get_dry_deposition_velocity()
        pm01r = pm01_model.get_pm(water, pm01_ddv)
        return (pm10r, pm25r, pm01r, pm10_ddv, pm25_ddv, pm01_ddv)

    def fluxes(self):
        """
        This method calculates:
            - VOC emissions
            - PM removal
            - O3 fluxes
            - NO2 fluxes
            - SO2 fluxes
            - CO fluxes

        Raises:
        NotImplementedError: If data is available for less than two days
        """
        atm3    = self.atm3
        canopy3 = self.canopy3
        rad3    = self.rad3
        res3    = self.res3
        eb3     = self.eb3

        # VOC EMISSIONS
        bioem = PollutantModel(
            self.atm3,
            self.canopy3,
            self.rad3,
            self.res3,
            self.eb3,
            self.lai_layer,
            self.frequency
        )
        activity_coeff3 = bioem.get_activity_coefficients()
        flux_voc = bioem.get_voc_emission(activity_coeff3)

        # FLUXES: O₃ NO₂ SO₂ CO
        flux_pollutant = bioem.get_pollutant_fluxes()

        o3_flux_tmp = -flux_pollutant["O3flux_sto"]
        o3conc_ppb = atm3["O3CONC"]
        [o3_perc, _, _] = self.get_pollutant_perc_uptake(
            res3["z_bl"], res3["rho_boundary"], o3conc_ppb, o3_flux_tmp
        )
        flux_pollutant["o3_perc"] = o3_perc

        # PARTICULATE MATTER
        i_pm10, i_pm25, i_pm01 = self.get_particulate_input(
            canopy3,
            atm3,
            res3,
            eb3,
            rad3
        )

        [
            pm10r,
            pm25r,
            pm01r,
            pm10_ddv,
            pm25_ddv,
            pm01_ddv
        ] = self.get_particulate_results(i_pm10, i_pm25, i_pm01)

        pm1_tot_flux = np.nansum(pm01r['pm_uptake_rate'], axis=1)
        pm2_5_tot_flux = np.nansum(pm25r['pm_uptake_rate'], axis=1)
        pm10_tot_flux = np.nansum(pm10r['pm_uptake_rate'], axis=1)

        flux_pm = {
            "pm10_perc"              :        pm10r['pm_perc'] ,
            "pm2_5_perc"             :        pm25r['pm_perc'] ,
            "pm1_perc"               :        pm01r['pm_perc'] ,
            "dry_dep_velocity_pm10"  :                pm10_ddv ,
            "dry_dep_velocity_pm2_5" :                pm25_ddv ,
            "dry_dep_velocity_pm1"   :                pm01_ddv ,
            "pm10_uptake_ug__m2"     :         pm10r['pm_a2c'] ,
            "pm10_uptake_rate"       : pm10r['pm_uptake_rate'] ,
            "pm2_5_uptake_ug__m2"    :         pm25r['pm_a2c'] ,
            "pm2_5_uptake_rate"      : pm25r['pm_uptake_rate'] ,
            "pm1_uptake_ug__m2"      :         pm01r['pm_a2c'] ,
            "pm1_uptake_rate"        : pm01r['pm_uptake_rate'] ,
            "pm1_tot_flux"           :            pm1_tot_flux ,
            "pm2_5_tot_flux"         :          pm2_5_tot_flux ,
            "pm10_tot_flux"          :           pm10_tot_flux ,
        }
        flux_pollutant.update(flux_pm)

        # OZONE FORMING POTENTIAL
        flux_pollutant["ozone_fp"] = bioem.get_ozone_forming_potential(
            flux_voc['all_iso_emis'],
            flux_voc['all_mnt_emis'],
        )

        flux_pollutant['tot_ozone_fp'] = np.nansum(flux_pollutant['ozone_fp'], axis=1)
        flux_pollutant['o3_net'] = flux_pollutant["O3flux_tot_sum"] + flux_pollutant['tot_ozone_fp']

        # STORE OUTPUT POLLUTANTS
        return (flux_voc, flux_pollutant)


    def get_pollutant_perc_uptake(
        self, z_bl: float, rho2: np.ndarray, conc_ppb: np.ndarray, flux: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the percent uptake of a pollutant in the boundary layer.

        Parameters:
        z_bl (float): The height of the boundary layer in meters for clear skies
        rho2 (np.ndarray): The density of the pollutant
            in micrograms per cubic meter.
        conc_ppb (np.ndarray): The concentration of the pollutant
            in parts per billion.
        flux (float): The pollutant flux in nanomoles
            per square meter per second.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - pollutant_percent_uptake (np.ndarray): The percent uptake
                of the pollutant.
            - conc_ug (np.ndarray): The concentration of the pollutant
                in micrograms per cubic meter.
            - flux_ug (np.ndarray): The pollutant flux in micrograms per hour.
        """

        row, col = rho2.shape
        app_repeat = np.repeat(conc_ppb, col)
        app_repeat = np.array(app_repeat)

        conc_ppb = app_repeat.reshape(row, col)

        conc_ug = conc_ppb * rho2
        bl = conc_ug * z_bl  # (ug O3 / m2 canopy)
        flux_ug = (flux / 1000) * 48 * 3600  # ug h-1

        pollutant_percent_uptake = (flux_ug / (flux_ug + bl)) * 100
        return pollutant_percent_uptake, conc_ug, flux_ug




class PollutantModel():
    def __init__(self, atm3, canopy3, rad3, res3, eb3, lai_layer, frequency):
        self.atm3 = atm3
        self.canopy3 = canopy3
        self.rad3 = rad3
        self.res3 = res3
        self.eb3 = eb3
        self.lai_layer = lai_layer
        self.frequency = frequency
        self.gamma_sm = np.expand_dims(eb3["SM"], 1)
        self.ce = 0.57  # Megan v2.1. TODO: change using standard conditions

    def get_cst(self, compound_class):
        match compound_class:
            case "monoterpenes":
                c_ct1 = 80
                c_ldf = 0.5  # mean for all monoterpenes
                c_beta = 0.1
            case "isoprene":
                c_ct1 = 95
                c_ldf = 1
                c_beta = 0.13
            case "sesquiterpenes":
                c_ct1 = 130
                c_ldf = 0.5
                c_beta = 0.17
            case "ovoc":
                # mean for acetone, methanol
                c_ct1 = 70
                c_ldf = 0.4
                c_beta = 0.09
            case _:
                raise ValueError(f"compound class {compound_class} not found")
        return (
            c_ct1,
            c_ldf,
            c_beta
        )

    def get_ozone_forming_potential(self, isoprene_emiss_sum, mt_emiss_sum):
        # ozone forming potential
        reactivity_factor_isoprene = 9.1  # g O3 g isoprene⁻¹
        reactivity_factor_monoterpenes = 3.8  # g O3 g monoterpenes⁻¹
        ozone_forming_potential = (
            isoprene_emiss_sum
            * reactivity_factor_isoprene
            + mt_emiss_sum
            * reactivity_factor_monoterpenes
        )
        return ozone_forming_potential

    def get_activity_coefficients(self):
        # atm3 = self.atm3
        rad3 = self.rad3
        eb3 = self.eb3
        canopy3 = self.canopy3

        leaf_temperature = eb3["leaf_temperature"]

        par = np.nan_to_num(np.array(rad3["par"]))
        par = np.tile(par, (canopy3.layers, 1)).T
        par_sun = rad3["par_sun"] * 4.17
        par_shade = rad3["par_shaded"] * 4.17

        # REPEAT DATA FOR 10 DAYS
        # _1:  data of the first day
        # _10:  data of the first 10 days
        # _ext1: data including repetition of the first day
        # _ext10: data including repetition of the first 10 days
        len_10 = 480 // self.frequency
        len_1 = len_10 // 10
        par = np.expand_dims(rad3["par"], axis=1)
        lft_1 = leaf_temperature[:len_1]  # leaf temperature
        par_1 = par[:len_1]              # par
        lft_10 = leaf_temperature[:len_10]  # leaf temperature
        par_10 = par[:len_10]              # par

        par_ext1 = np.concatenate([par_1, par])
        lft_ext1 = np.concatenate([lft_1, leaf_temperature])
        par_ext10 = np.concatenate([par_10, par])
        lft_ext10 = np.concatenate([lft_10, leaf_temperature])

        # compute mobile means
        p24 = moving_average(par_ext1, len_1)
        t24 = moving_average(lft_ext1, len_1)
        p240 = moving_average(par_ext10, len_10)
        t240 = moving_average(lft_ext10, len_10)

        c_ct2 = 230
        c_d1  = 0.004
        c_d2  = 0.0005
        c_d3  = 0.0468
        c_d4  = 2.034
        c_d5  = 0.05
        c_d6  = 313
        c_d7  = 0.6
        c_p0  = 200
        c_t0  = 297

        (
            mnt_ct1,
            mnt_ldf,
            mnt_beta
        ) = self.get_cst("monoterpenes")

        (
            iso_ct1,
            iso_ldf,
            iso_beta
        ) = self.get_cst("isoprene")

        (
            sqt_ct1,
            sqt_ldf,
            sqt_beta
        ) = self.get_cst("sesquiterpenes")
        (
            ovoc_ct1,
            ovoc_ldf,
            ovoc_beta
        ) = self.get_cst("ovoc")

        # fix to avoid log(0) error
        p24[p24 == 0] = 0.00000000000001
        p240[p240 == 0] = 0.00000000000001

        # gamma light dependent
        def get_gamma_p(par):
            # Guenther et al. 2012 (eq. 4)

            gamma = (c_d3 * np.exp(c_d2 * (p24 - c_p0)) * (p240) ** 0.6) * (
                ((c_d1 - c_d2 * np.log(p240)) * par)
                / ((1 + ((c_d1 - c_d2 * np.log(p240)) ** 2) * (par**2)) ** 0.5)
            )
            return gamma
        gamma_p_sun = get_gamma_p(par_sun)
        gamma_p_shade = get_gamma_p(par_shade)

        # gamma temperature dependent
        def get_gamma_t(c_ct1):
            gamma_t = (
                c_d4 * np.exp(
                    c_d5 * (t24 - c_t0)
                )
                * np.exp(c_d5 * (t240 - c_t0))) * (
                    c_ct2 * np.exp(
                        c_ct1 * (((1.0 / (c_d6 + (c_d7 * (t240 - c_t0))))
                                  - (1.0 / leaf_temperature)) / 0.00831)
                    ) / (c_ct2 - c_ct1
                         * (1 - np.exp(
                            c_ct2 * (((
                                1 / (c_d6 + (c_d7 * (t240 - c_t0)))
                            ) - (1.0 / leaf_temperature)) / 0.00831))))
            )
            return gamma_t

        mnt_gamma_t = get_gamma_t(mnt_ct1)
        iso_gamma_t = get_gamma_t(iso_ct1)

        # Guenther et al. 2012 (eq. 3)
        def apply_ldf(gamma_p, gamma_t, ldf):
            # TODO: check gamma_lai
            return (
                (1 - ldf) * gamma_t + ldf * gamma_p * gamma_t
            )  * self.gamma_sm * self.ce

        mnt_sn_gamma_pt  = apply_ldf(gamma_p_sun, mnt_gamma_t, mnt_ldf)
        mnt_sh_gamma_pt  = apply_ldf(gamma_p_shade, mnt_gamma_t, mnt_ldf)
        iso_sn_gamma_pt  = apply_ldf(gamma_p_sun, iso_gamma_t, iso_ldf)
        iso_sh_gamma_pt  = apply_ldf(gamma_p_shade, iso_gamma_t, iso_ldf)

        mnt_gamma_t_g95  = np.exp(
            mnt_beta * (np.nan_to_num(leaf_temperature - 303))
        )
        iso_gamma_t_g95  = np.exp(
            iso_beta * (np.nan_to_num(leaf_temperature - 303))
        )
        sqt_gamma_t_g95  = np.exp(
            sqt_beta * (np.nan_to_num(leaf_temperature - 303))
        )
        ovoc_gamma_t_g95 = np.exp(
            ovoc_beta * (np.nan_to_num(leaf_temperature - 303))
        )

        output = {
            'mnt_sn_gamma_pt' :   mnt_sn_gamma_pt,
            'mnt_sh_gamma_pt' :   mnt_sh_gamma_pt,
            'iso_sn_gamma_pt' :   iso_sn_gamma_pt,
            'iso_sh_gamma_pt' :   iso_sh_gamma_pt,
            'mnt_gamma_t_g95' :   mnt_gamma_t_g95,
            'iso_gamma_t_g95' :   iso_gamma_t_g95,
            'sqt_gamma_t_g95' :   sqt_gamma_t_g95,
            'ovoc_gamma_t_g95':   ovoc_gamma_t_g95,
        }
        return output

    def get_voc_emission(self, activity_coeff3):
        lai_layer = self.lai_layer

        # --> nmol m-2 s-1
        ef_monoterpenes_lt = self.canopy3["ef_nmol_monoterpenes_lt"]
        ef_monoterpenes_t  = self.canopy3["ef_nmol_monoterpenes_t"]
        ef_isoprene        = self.canopy3["ef_nmol_isoprene"]
        ef_sqt             = self.canopy3["ef_nmol_sesquiterpenes"]

        # <-- ug g-1 h-1
        ef_ovoc = self.canopy3["ef_ovoc"]

        fraction_sun_leaves = np.nan_to_num(np.array(
            self.rad3["fraction_sun_leaves"]
        ))

        mnt_emis_g95 = (
            ef_monoterpenes_t * activity_coeff3['mnt_gamma_t_g95'] * lai_layer
        )

        mnt_emis_pt = (
            ef_monoterpenes_lt * activity_coeff3['mnt_sn_gamma_pt']
            * fraction_sun_leaves * lai_layer
            + ef_monoterpenes_lt * activity_coeff3['mnt_sh_gamma_pt']
            * (1 - fraction_sun_leaves) * lai_layer
        )
        all_mnt_emis = mnt_emis_g95 + mnt_emis_pt

        all_iso_emis = (
            ef_isoprene * activity_coeff3['iso_sn_gamma_pt']
            * fraction_sun_leaves * lai_layer
            + ef_isoprene * activity_coeff3['iso_sh_gamma_pt']
            * (1 - fraction_sun_leaves) * lai_layer
        )

        all_sqt_emis = (
            ef_sqt * activity_coeff3['sqt_gamma_t_g95'] * lai_layer
        )

        # ug g-1 h-1
        all_ovoc_emis_ug_g = (
            ef_ovoc * activity_coeff3['ovoc_gamma_t_g95'] * lai_layer
        )

        return {
            'mnt_emis_g95' : mnt_emis_g95,
            'mnt_emis_pt' : mnt_emis_pt,
            'all_mnt_emis' : all_mnt_emis,
            'all_iso_emis' : all_iso_emis,
            'all_sqt_emis' : all_sqt_emis,
            'all_ovoc_emis_ug_g' : all_ovoc_emis_ug_g,
        }

    def get_pollutant_fluxes(self):
        eb3     = self.eb3
        res3    = self.res3
        atm3    = self.atm3
        canopy3 = self.canopy3

        gs_leaf_m_s_all_ozone = eb3["all_Gs_leaf_m_s"] * 0.61
        res_sto_ozone = 1 / gs_leaf_m_s_all_ozone
        rg_ozone = 200  # % s m⁻¹ from fares et al.
        res_mes_ozone = 10  # not revelant

        res_sto_ozone = res_mes_ozone + res_sto_ozone
        rc_o3 = (
            1 / res_sto_ozone + 1 / res3["cuticolar_resistance_O3"]
        ) ** (-1)

        res_tot_o3 = (
            res3["atmospheric_resistance"]
            + res3["boundary_layer_res_O3"]
            + rc_o3
        )

        vd_o3_tot_canopy = 1 / (res_tot_o3)

        ozone = np.tile(atm3["O3CONC"], (int(canopy3["layers"]), 1))
        ozone = ozone.transpose()

        res_atmospheric_resistance = np.array(res3["atmospheric_resistance"])
        res_boundary_layer_res_o3 = np.array(res3["boundary_layer_res_O3"])

        o3c = (ozone * rc_o3) / (
            res_atmospheric_resistance + res_boundary_layer_res_o3 + rc_o3
        )
        eb_rho_mol_2d = np.expand_dims(
            eb3["rho_mol"],
            axis=1
        )  # required 2 dim arrays
        eb_rho_mol_1d = eb3["rho_mol"]  # for 1 dim arrays
        o3flux_sto = -o3c * eb_rho_mol_2d * (1 / res_sto_ozone)
        o3flux_sto_sum = np.sum(o3flux_sto, 1)

        o3flux_canopy = -ozone * eb_rho_mol_2d * vd_o3_tot_canopy
        o3flux_canopy_sum = np.sum(o3flux_canopy, 1)

        o3flux_soil = (-o3c[:, -1] * eb_rho_mol_1d) / (
            res3["in_canopy_atm_resistance"][:, -1] + rg_ozone
        )
        o3flux_tot = np.column_stack((o3flux_canopy, o3flux_soil))
        o3flux_tot_sum = np.sum(o3flux_tot, 1)

        if np.isnan(sum(atm3["NO2"])):
            no2flux_tot_sum = []
        else:
            rg_no2 = 200  # s m⁻¹ from fares et al. SR
            res_sto_no2 = res_sto_ozone
            res_mes_no2 = 600  # % Novak
            res_sto_no2 = res_mes_no2 + res_sto_no2

            res_cuticolar_resistance_no2 = res3["cuticolar_resistance_NO2"]
            rc_no2 = (
                1 / res_sto_no2 + 1 / res_cuticolar_resistance_no2
            ) ** (-1)

            res_tot_no2 = (
                res3["atmospheric_resistance"]
                + res3["boundary_layer_res_NO2"]
                + rc_no2
            )
            vd_no2_tot_canopy = 1 / (res_tot_no2)

            app_no2 = np.tile(atm3["NO2"], (int(canopy3["layers"]), 1))
            no2 = app_no2.transpose()
            no2flux_canopy = -no2 * eb_rho_mol_2d * vd_no2_tot_canopy
            no2c = (no2 * rc_no2) / res_tot_no2

            no2flux_soil = (-no2c[:, -1] * eb_rho_mol_1d) / (
                res3["in_canopy_atm_resistance"][:, -1] + rg_no2
            )
            no2flux_tot = np.column_stack((no2flux_canopy, no2flux_soil))
            no2flux_tot_sum = np.sum(no2flux_tot, 1)

        if np.isnan(sum(atm3["SO2"])):
            so2flux_tot_sum = []
        else:
            gs_leaf_m_s_all_so2 = eb3["all_Gs_leaf_m_s"] * (0.12 / 0.22)  # OK
            res_sto_so2 = 1 / gs_leaf_m_s_all_so2
            rg_so2 = 2941  # Novak
            res_mes_so2 = 0  # (Wesely 1989)
            res_sto_so2 = res_mes_so2 + res_sto_so2
            rc_so2 = (
                1 / res_sto_so2 + 1 / res3["cuticolar_resistance_SO2"]
            ) ** (-1)
            res_tot_so2 = (
                res3["atmospheric_resistance"]
                + res3["boundary_layer_res_SO2"]
                + rc_so2
            )
            vd_so2_tot_canopy = 1 / (res_tot_so2)

            app_so2 = np.tile(atm3["SO2"], (int(canopy3["layers"]), 1))
            so2 = app_so2.transpose()
            so2flux_canopy = -so2 * eb_rho_mol_2d * vd_so2_tot_canopy
            so2c = (so2 * rc_so2) / (
                res3["atmospheric_resistance"]
                + res3["boundary_layer_res_SO2"]
                + rc_so2
            )
            so2flux_soil = (-so2c[:, -1] * eb_rho_mol_1d) / (
                res3["in_canopy_atm_resistance"][:, -1] + rg_so2
            )
            so2flux_tot = np.column_stack((so2flux_canopy, so2flux_soil))
            so2flux_tot_sum = np.sum(so2flux_tot, 1)

        if np.isnan(sum(atm3["CO"])):
            co_flux_tot_sum = []
        else:
            rc_co = 50000  # % s m⁻¹
            rg_co = np.repeat(2000, len(atm3["CO"]))
            idx = np.where((atm3["TOY"] > 60.0) & (atm3["TOY"] < 180.0))
            rg_co[idx] = 2941  # s m⁻¹
            res_tot_co = (
                res3["atmospheric_resistance"]
                + res3["boundary_layer_res_CO"]
                + rc_co
            )
            vd_co_tot_canopy = 1 / (res_tot_co)

            app_co = np.tile(atm3["CO"], (int(canopy3["layers"]), 1))
            carbon_monoxide = app_co.transpose()
            co_flux_canopy = -carbon_monoxide * eb_rho_mol_2d * vd_co_tot_canopy
            co_c = (carbon_monoxide * rc_co) / (
                res3["atmospheric_resistance"]
                + res3["boundary_layer_res_CO"]
                + rc_co
            )
            co_flux_soil = (-co_c[:, -1] * eb_rho_mol_1d) / (
                res3["in_canopy_atm_resistance"][:, -1] + rg_co
            )
            co_flux_tot = np.column_stack((co_flux_canopy, co_flux_soil))
            co_flux_tot_sum = np.sum(co_flux_tot, 1)
        flux_pollutant = {
            "RstoO3"            :  res_sto_ozone     ,
            "O3c"               :  o3c               ,
            "VdO3_tot_canopy"   :  vd_o3_tot_canopy  ,
            "O3flux_tot_sum"    :  o3flux_tot_sum    ,
            "O3flux_tot"        :  o3flux_tot        ,
            "O3flux_soil"       :  o3flux_soil       ,
            "O3flux_canopy_sum" :  o3flux_canopy_sum ,
            "O3flux_canopy"     :  o3flux_canopy     ,
            "O3flux_sto_sum"    :  o3flux_sto_sum    ,
            "O3flux_sto"        :  o3flux_sto        ,
            "COflux_tot_sum"    :  co_flux_tot_sum   ,
            "COflux_tot"        :  co_flux_tot       ,
            "SO2flux_tot"       :  so2flux_tot       ,
            "SO2flux_tot_sum"   :  so2flux_tot_sum   ,
            "NO2flux_tot_sum"   :  no2flux_tot_sum   ,
            "NO2flux_tot"       :  no2flux_tot       ,
        }
        return flux_pollutant


@dataclass
class InputParticulate:
    '''
    lai:                  [m² m⁻²] leaf area index
    dry_deposition:       [μg m⁻²] dry deposition on the surface of foliage
    rain:                 [mm] water precipitation
    wsp:                  [m s⁻¹] wind speed at the top of the canopy
    hcl:                  [m] canopy layer height
    htop:                 [m] canopy total height
    aereodynamic_resistance: [s m⁻¹] aereodynamic resistance

    input for get_pm_dep_velocity:
        particulate_diameter: [m] diameter of PM
        ustar:                [m s⁻¹] friction velocity
        leaf_width:           [m] leaf width
        air_temperature:      [K] air temperature

    input for get_pm:
        z_bl:                 [m] boundary layer height
    '''
    rain:                          np.ndarray
    lai:                           np.ndarray
    wsp:                           np.ndarray
    hcl:                           np.ndarray

    mean_wind_velocity:            np.ndarray
    ustar:                         np.ndarray
    air_temperature:               np.ndarray

    aereodynamic_resistance:       np.ndarray
    aereodynamic_roughness_height: np.ndarray

    hsn_lh_evap :                   np.ndarray
    hsd_lh_evap :                   np.ndarray
    fraction_sun_leaves:            np.ndarray

    htop:       float
    leaf_width: float
    leaf_length: float
    leaf_thickness: float
    z_bl:       float
    broadleaf:  bool

    # assigned in setup method
    particulate_diameter: float = np.nan
    particulate_density: float = np.nan
    particulate_conc: np.ndarray = field(default_factory=lambda: np.zeros(1))


    def setup(
        self,
        particulate_diameter: float,
        particulate_conc: np.ndarray,
        particulate_density: float,
    ):
        '''
            finalize init data
            particulate_diameter: [m] diameter of PM
            particulate_conc:     [μg m⁻³] concentration of PM
        '''
        self.particulate_diameter = particulate_diameter
        self.particulate_conc     = particulate_conc
        self.particulate_density  = particulate_density


class ParticulateModel:
    def __init__(self, inp: InputParticulate):
        self.inp = inp

        shp = self.inp.lai.shape
        assert len(shp) == 2
        self.shp = shp
        self.len_ts = inp.lai.shape[0]     # time series length
        self.len_layer = inp.lai.shape[1]  # number of layers

    def get_water(self):
        '''
        return dict with:
            pm_washoff_ratio:       [-] pm washoff ratio for each canopy layers
            water_layer:            [mm] water amount over each layer
            water_drip:             [mm] water drip from each layer
            water_storage_capacity: [mm] water storage capacity for each layer
            interception_fraction:  [-] interception fraction for each layer

        input:
            rain: [mm] water precipitation
            lai: [m² m⁻²] leaf area index
            evaporation: TODO [mm] water evaporation from the surface of foliage

        note:
            canopy is imposed dry at the beginning of the simulation

        ref:
            Schaubroeck et al. 2014
        '''
        rain = self.inp.rain
        lai  = self.inp.lai

        # TODO: assess evaporation with @silvano
        # water enthalpy of vaporization

        '''
        Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop evapotranspiration - Guidelines for computing crop water requirements. FAO Irrigation and drainage Paper 56.
        '''
        # todo: temperature in °C leaf_temperature
        # todo: water condensation
        # todo: reference
        t_k = self.inp.air_temperature
        conversion_factor = 0.00027777777777778  # from [J kg⁻¹] to [watt hour⁻¹ mm⁻¹]
        latent_heat_of_vaporization = np.expand_dims(
            conversion_factor
            * 1.91846 * 1e6 * (t_k / (t_k - 33.91)) ** 2
            , 1
        )
        time_step_hours = 3
        hsn_lh_evap = self.inp.hsn_lh_evap * self.inp.fraction_sun_leaves
        hsd_lh_evap = self.inp.hsd_lh_evap * (1 - self.inp.fraction_sun_leaves)
        lh_evap = hsn_lh_evap + hsd_lh_evap

        # TODO: check if PM cycle can be partitioned in sun and shade leaves
        evaporation = (
            lh_evap * time_step_hours / latent_heat_of_vaporization
        )

        # [-] (forest) from Xiao & McPherson 2016
        interception_coefficient = 0.7

        # [mm] specific water storage capacity == 0.2 in Itree
        specific_water_storage_capacity = 0.86  # [mm]  Xiao & McPherson 2016

        # [mm] (eq. 5) storage capacity  # TODO: check with @silvano
        storage_capacity = lai * 2 * specific_water_storage_capacity

        # [-] interception fraction (eq.3) from: Deckmyn et al. 2008
        interception_fraction = 1 - np.exp(-interception_coefficient * lai)

        water_input      = np.zeros(self.shp)
        water_drip       = np.zeros(self.shp)
        water_layer      = np.zeros(self.shp)
        pm_washoff_ratio = np.zeros(self.shp)

        for its in range(self.len_ts):
            for ilr in range(self.len_layer):
                # water input [mm] (eq. 4)
                ratio = interception_fraction[its, ilr]
                n_ratio = 1 - ratio

                if ilr == 0:
                    water_input[its, ilr] = rain[its]
                else:
                    # alessandro: eq. 4 modified:
                    # only part of water drip is intercepted
                    water_input[its, ilr] = (
                        (water_input[its, ilr  - 1]
                         + water_drip[its, ilr  - 1])
                        * n_ratio
                    )

                # water on layer before drip [mm]
                wl0 = (
                    water_input[its, ilr] * ratio
                    - evaporation[its, ilr]
                )
                if its > 0:
                    # add water from previous time step
                    wl0 += water_layer[its  - 1, ilr]
                wl0 = max(wl0, 0)

                # water drip [mm] (eq. 9)
                wd1 = wl0 - storage_capacity[its, ilr]
                water_drip[its, ilr] = max(wd1, 0)

                # water amount of layer [mm] (eq. 2)
                water_layer[its, ilr] = wl0 - water_drip[its, ilr]


        # eq. 17 (part)
        pm_washoff_ratio = water_drip / (water_drip + storage_capacity)
        return {
                'pm_washoff_ratio':       pm_washoff_ratio,
                'water_layer':            water_layer,
                'water_drip':             water_drip,
                'water_storage_capacity': storage_capacity,
                'interception_fraction':  interception_fraction,
        }

    def get_dry_deposition_velocity(
        self,
    ):
        """
        Calculates the deposition velocities of particulate matter

        Parameters:
            particulate_diameter: [m] diameter of PM
            ustar:                [m s⁻¹] friction velocity
            leaf_width:           [m] leaf width
            particulate_density:  [kg m⁻³] density of PM (ref: 2e3 Zhang 2001)
            air_temperature:      [K] air temperature
            broadleaf:            [bool] True if broadleaf, False if conifer

        Returns dict with:
            v_dry_deposition: [m s⁻¹] deposition velocity
            v_interception:   [m s⁻¹] interception velocity
            v_impaction:      [m s⁻¹] impaction velocity
            v_settling:       [m s⁻¹] gravitational settling velocity

        Reference:
            Liu et al. 2016
        """
        particulate_diameter = self.inp.particulate_diameter
        ustar                = self.inp.ustar
        air_temperature      = self.inp.air_temperature
        particulate_density  = self.inp.particulate_density

        air_temperature = np.expand_dims(air_temperature, axis=1)

        # Characteristic radius (r) is a vague term
        # For an arbitrary shape with volume V = 4/3 * pi * r^3

        area_correction_factor = 2 / 3
        leaf_area = (
                self.inp.leaf_width
                * self.inp.leaf_length
                * area_correction_factor
        )

        # characteristic length of a collector (Poulet et al. 2022)
        characteristic_length = 2 * (
                leaf_area
                / np.pi
        ) ** 0.5
        #characteristic_radius  = self.inp.leaf_width + self.inp.leaf_length) / 2


        # parameters for impaction efficiency
        # [m] (broadleaves) Pleim et al. 2022 (table 1)
        if self.inp.broadleaf:
            width_micro = 1e-6
        else:
            width_micro = 0.5e-6
        f_micro = 0.8 / 100  # [-] Pleim et al. 2022

        # air dynamic viscosity
        # [Pa s == kg m⁻¹ s⁻¹] Sutherland 1893
        cc1 = 1.458e-6  # [kg m⁻¹ s⁻¹ K^(1/2)]
        cc2 = 110.4  # [K]
        air_viscosity = cc1 * air_temperature**(3 / 2) / (air_temperature + cc2)

        # [m] Liu et al. 2016
        free_air_path = 65e-9

        # [-] Liu et al. 2016
        cunningham_factor = 1 + ((2 * free_air_path) / particulate_diameter) * (
            1.257 + 0.4 * np.exp(
                -0.55 * particulate_diameter / free_air_path
            )
        )

        # [s] eq. 19
        # (kg m⁻³) * () * (m²) / (kg m⁻¹ s⁻¹) = (s)
        particle_relaxation_time = (
            particulate_density * cunningham_factor * (particulate_diameter**2)
        ) / (18 * air_viscosity)
        particle_relaxation_time = np.expand_dims(
            particle_relaxation_time,
            axis=1
        )

        # gravitational settling velocity -->
        # [-] Zhang 2001, eq. 3
        correction_factor = (
            1
            + (2 * free_air_path / particulate_diameter)
            * (1.257 + 0.4 * np.exp(
                -0.55 * particulate_diameter / free_air_path))
        )

        # [m s⁻¹] Zhang 2001, eq. 2
        # (kg m⁻³) * (m²) * (m s⁻²) / (kg m⁻¹ s⁻¹) = (m s⁻¹)
        gravitational_settling_velocity = (
            particulate_density
            * cunningham_factor
            * particulate_diameter**2
            * 9.8
            * correction_factor
            / (18 * air_viscosity)
        )

        # diffusion efficiency -->
        # Petroff et al. 2008
        boltzmann_constant = 1.38e-23  # [m² kg s⁻² K⁻¹]
        browsian_diffusion_coefficient = (
            boltzmann_constant
            * air_temperature
            * cunningham_factor
            / (3 * np.pi * air_viscosity * particulate_diameter)
        )

        # Pleim et al. 2022
        ratio_viscious_drag_to_total = 1 / 3  # (grass) Chamberlain 1967
        schmidt_number = air_viscosity / browsian_diffusion_coefficient
        efficiency_diffusion = (
            ratio_viscious_drag_to_total
            * schmidt_number**(-2 / 3)
        )

        # impaction efficiency -->
        # [-] Slinn 1982 eq. 29
        stokes_number_leaf = (
            gravitational_settling_velocity
            * ustar
            / (9.8 * characteristic_length)
        )
        stokes_number_micro = (
            gravitational_settling_velocity
            * ustar
            / (9.8 * width_micro)
        )
        efficiency_impaction = (
            (1 - f_micro)
            * stokes_number_leaf ** 2
            / (1 + stokes_number_leaf ** 2)
            + f_micro
            * stokes_number_micro ** 2
            / (1 + stokes_number_micro ** 2)
        )
        # <--

        # interception efficiency
        # [-] eq. 25
        efficiency_interception = (
            0.5 * particulate_diameter / characteristic_length
        )

        # rebound
        # [-] eq. 24
        rebound = np.exp(-(stokes_number_leaf**0.5))
        # in Liu et al. 2016 (under eq. 23)
        if particulate_diameter < 5e-6:
            rebound = 1

        # quasi-laminar boundary layer resistance
        # Pleim et al 2022, from Slinn 1982
        qlbl_resistance = (
            1 / (
                self.inp.lai * ustar
                * (
                    efficiency_diffusion
                    + efficiency_impaction
                    + efficiency_interception
                )
            )  * rebound
        )

        # dry deposition velocity
        ddv = (
            gravitational_settling_velocity / (1 - np.exp(
                - gravitational_settling_velocity
                * (qlbl_resistance + self.inp.aereodynamic_resistance)
            ))
        )
        return ddv

    def get_pm(
        self,
        water3: dict,
        dry_deposition_velocity: np.ndarray
    ):
        '''

        input:
            self.inp.hcl: (m) midpoint height of canopy layer
            self.inp.htop: (m) totale height of the tree
            self.inp.dry_deposition: (μg m⁻²) dry deposition over the layer
            water3: output from get_water_cycle method
            dry_deposition_velocity: output from get_deposition_velocities method

            evaporation: TODO (mm) water evaporation from the surface of foliage

        return dict with:

        note:
            canopy is supposed dry at the beginning of the simulation

        ref:
            Schaubroeck et al. 2014
        '''
        # TODO: include leaf turnover

        #asperity_diameter = 0.001
        # WRF model default (urban parks) Shen et al. 2023
        asperity_diameter = 0.5

        # asperity_diameter = self.inp.aereodynamic_roughness_height


        conc = np.expand_dims(self.inp.particulate_conc, 1)

        # [μg m⁻² s⁻¹]
        # (m s⁻¹) * (μg m⁻³) = (μg m⁻² s⁻¹)
        downward_deposition_flux = (
            dry_deposition_velocity
            * conc
        )

        # [μg m⁻²] in the interval of time
        time_interval = 60 * 30 * 6  # [s]
        pm_dry_deposition = downward_deposition_flux * time_interval  # [μg m⁻²]

        # [s⁻¹] Loosmore 2003; Henry and Minier 2014
        # dimensionality check not carried out: empirical equation
        resuspension_rate = (
            0.42 * self.inp.ustar**2.13
            * self.inp.particulate_diameter**0.17
            / (
                time_interval**0.92
                * asperity_diameter**0.32
                * self.inp.particulate_density**0.76
            )
        )

        # model 1
        #resuspension_rate = 0.01 * ( self.inp.ustar**1.43) / time_interval**1.03

        # resuspension_rate_nowak = self.get_resuspension_ratio_nowak(self.inp.mean_wind_velocity)

        # --> init arrays
        # [μg m⁻²] PM input that may be intercepted by the canopy layers
        pm_input        = np.zeros(self.shp) # dry + wet
        # [μg m⁻²] PM on the surface of foliage
        pm_leaf         = np.zeros(self.shp)
        # [μg m⁻²] delta of PM between time steps
        pm_delta_leaf   = np.zeros(self.shp)
        # [μg m⁻²] resuspension of PM
        pm_resuspension = np.zeros(self.shp)
        # [μg m⁻²] washoff of PM
        pm_washoff      = np.zeros(self.shp)
        pm_leaf_fall    = np.zeros(self.shp)

        resuspension_factor = np.zeros(self.shp)
        resuspension_ratio = np.zeros(self.shp)

        ifr = water3['interception_fraction']
        nfr = 1 - ifr
        ofr: list[np.ndarray] = []
        mat = np.zeros(self.shp)
        mat[:, 0] = 1
        ofr.append(mat)
        for ilr in range(2, self.len_layer):
            mat = np.zeros(self.shp)
            for ix1 in range(1, ilr):
                mat[:, ix1] = np.prod(nfr[:, ix1:ilr], axis=1)
            mat[:, ix1 + 1] = 1
            mat = np.delete(mat, 0, axis=1)
            zeros = np.expand_dims(np.zeros(self.shp[0]), axis=1)
            mat = np.append(mat, zeros, axis=1)
            ofr.append(mat)

        # [-] coefficient for pm_deposition
        ci_pm = self.inp.lai

        for its in range(self.len_ts):
            for ilr in range(self.len_layer):

                if its == 0:
                    # no resuspension or washoff at first time step
                    pm_input[its, ilr] = pm_dry_deposition[its, ilr] * ci_pm[its, ilr]
                    pm_delta_leaf[its, ilr] = pm_input[its, ilr]
                    pm_leaf[its, ilr] = pm_delta_leaf[its, ilr]
                    continue
                # [μg m⁻²] eq. 15
                if ilr != 0:
                    # particulate input for each layer starting from 2nd
                    # [μg m⁻²] eq. 13
                    pm_input[its, ilr] = (
                        pm_dry_deposition[its, ilr] * ci_pm[its, ilr]
                        + np.sum(pm_washoff[its - 1, :]
                                 * ofr[ilr - 1][its, :])
                        * ifr[its, ilr]
                    )
                else:
                    pm_input[its, ilr] = pm_dry_deposition[its, ilr] * ci_pm[its, ilr]

                pm_washoff[its - 1, ilr] = (
                    pm_leaf[its - 1, ilr]
                    * water3['pm_washoff_ratio'][its - 1, ilr]
                )

                # --> modelling particulate to the ground for leaf fall
                pm_leaf_fall[its, ilr] = 0
                lai_time_ratio = min(
                    ci_pm[its, ilr] / ci_pm[its - 1, ilr],
                    1
                )
                pm_leaf_fall[its, ilr] = (
                    (pm_leaf[its - 1, ilr] - pm_washoff[its - 1, ilr])
                    * (1 - lai_time_ratio)
                )

                # [μg m⁻²] eq. 16
                resuspension_factor[its - 1, ilr] = (
                    resuspension_rate[its - 1, ilr]
                    * (
                        1 - water3['water_layer'][its - 1, ilr]
                        / water3['water_storage_capacity'][its - 1, ilr]
                    )
                )
                res_ratio = resuspension_factor[its - 1, ilr]
                pm_leaf0 = (
                    pm_leaf[its - 1, ilr]
                    - pm_washoff[its - 1, ilr]
                    - pm_leaf_fall[its, ilr]
                )
                # coumpund negative interest
                pm_leaf1 = pm_leaf0 * np.power(1 - res_ratio, time_interval)
                pm_resuspension[its - 1, ilr] = pm_leaf0 - pm_leaf1
                resuspension_ratio[its - 1, ilr] = (
                    (pm_leaf0 - pm_leaf1) / pm_leaf0
                )
                # <--

                # [μg m⁻²] eq. 12
                pm_delta_leaf[its, ilr] = (
                    pm_input[its, ilr]
                    - pm_washoff[its - 1, ilr]
                    - pm_leaf_fall[its, ilr]
                    - pm_resuspension[its - 1, ilr]
                )

                # [μg m⁻²] pm leaf
                pm_leaf[its, ilr] = (
                    pm_leaf[its - 1, ilr]
                    + pm_delta_leaf[its, ilr]
                )

        # [-] fraction not intercepted by layers below
        cnot_intercepted_frac = np.zeros(self.shp)
        for ilr in range(1, self.len_layer):
            not_intercept = np.prod(nfr[:, ilr:], axis=1)
            cnot_intercepted_frac[:, ilr] = not_intercept
        ones = np.expand_dims(np.ones(self.shp[0]), axis=1)
        cnot_intercepted_frac = np.delete(
            cnot_intercepted_frac,
            0,
            axis=1
        )
        cnot_intercepted_frac = np.append(
            cnot_intercepted_frac,
            ones,
            axis=1
        )

        # --> PM removal computation
        # computed differently from ref
        # [μg m⁻²] PM transfer from canopy to ground (washoff to the ground)
        pm_c2g = (
            pm_washoff
            * cnot_intercepted_frac
            + pm_leaf_fall
        )

        # [μg m⁻²] PM transfer between atmosphere and canopy
        # pm_a2c.sum() == pm_c2g.sum() + pm_leaf[-1, :].sum()
        pm_a2c = (
            pm_dry_deposition
            * ci_pm
            - pm_resuspension
        )

        # final computations -->
        # [μg m⁻² s⁻¹]
        pm_uptake_rate = pm_a2c / time_interval

        # [μg m⁻²] in the interval of time
        # TODO discuss with @silvano (no buono)
        pm_bl = self.inp.z_bl * np.expand_dims(self.inp.particulate_conc, 1)
        pm_perc = (
            100 * pm_a2c
            / (pm_a2c + pm_bl)
        )

        # print("pm_a2c (g)", pm_a2c.sum() * 1e-6)
        # print("pm_c2g (g)", pm_c2g.sum() * 1e-6)
        # print("pm_leaf[-1, :]", pm_leaf[-1, :].sum() * 1e-6)
        # print("resuspension ratio: ", pm_resuspension.sum() / (pm_dry_deposition * ci_pm).sum())

        return {
            'pm_delta_leaf': pm_delta_leaf,
            'pm_leaf': pm_leaf,
            'pm_washoff': pm_washoff,
            'pm_c2g': pm_c2g,
            'pm_a2c': pm_a2c,
            'pm_uptake_rate': pm_uptake_rate,
            'pm_perc': pm_perc,
        }

    def get_resuspension_ratio_nowak(self, wind_speed: np.ndarray):
        # Nowak et al. 2013 - Table 3
        # index is the wind speed in m/s
        resuspention_percent = np.array([0, 1.5, 3, 4.5, 6, 7.5, 9,
                                10, 11, 12, 13, 16, 20, 23])
        resuspesion_rate = resuspention_percent / 100
        ix_low = wind_speed.astype(int)

        # resuspension rate (-)
        rr = np.zeros(self.shp)

        # resuspension rate is constant for wind speed > 13 m/s
        ix_max = ix_low >= 13
        rr[ix_max] = resuspesion_rate[-1]

        # linear interpolation for wind speed < 13 m/s
        ix_high = ix_low + 1
        diff1 = (wind_speed - ix_low)[~ix_max]
        rr_high = np.array(list(map(lambda x: resuspesion_rate[x], ix_low[~ix_max])))
        rr_low =  np.array(list(map(lambda x: resuspesion_rate[x], ix_high[~ix_max])))
        rr[~ix_max] = rr_low + (rr_high - rr_low) * diff1  # rr_high - rr_low = 1 always
        return rr


def moving_average(arr, window_size):
    # TODO: check
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array t o
    #consider every window of size 3
    while i < len(arr) - window_size:

        # Calculate the average of current window
        #window_average = round(np.sum(arr[
        #  i:i+window_size]) / window_size, 2)
        window_average = np.mean(arr[i:i+window_size, :], axis=0)

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1

    return np.array(moving_averages)

