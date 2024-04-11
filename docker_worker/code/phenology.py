from enum import Enum
import ephem          # type: ignore
import polars as plr  # type: ignore
import numpy as np


class CorrectionType(Enum):
    'Enum for correction type'
    NONE = 0
    RAW = 1
    GSI = 2


class Phenology:
    # TODO: check for simulations longer than 1 year

    def __init__(
        self,
        is_deciduous: bool,
        latitude: float,
        longitude: float,
        doy_leaf_on: int,
        doy_leaf_off: int,
        atm3 : plr.DataFrame,
        frequency: int = 6,
    ):
        self.is_deciduous = is_deciduous
        self.latitude  = latitude
        self.longitude = longitude
        self.frequency = frequency
        self.gsi_threshold = 0.1
        self.minimum_days_to_apply_gsi_correction = 300
        days = len(atm3["TIMEs"]) / (24 / (0.5 * self.frequency))
        self._correction_type = CorrectionType.NONE
        self.correction_factor = 1.8  # gsi correction factor
        if self.is_deciduous:
            if days >= self.minimum_days_to_apply_gsi_correction:
                self._correction_type = CorrectionType.GSI
            else:
                self._correction_type = CorrectionType.RAW

        # compute phenology
        match(self._correction_type):
            case CorrectionType.NONE:
                pass
            case CorrectionType.RAW:
                self.is_veg = np.ones(len(atm3["TIMEs"]))
                toy2 = np.array(atm3["TOY"])
                self.is_veg[toy2 < doy_leaf_on]  = 1e-10
                self.is_veg[toy2 > doy_leaf_off] = 1e-10
            case CorrectionType.GSI:
                self._build_gsi(atm3)
            case _:
                raise ValueError(
                    f"Unknown correction type: {self._correction_type}"
                )

    def apply_correction(self, lai: float, ts_index: int) -> float:
        lai_out = lai
        match(self._correction_type):
            case CorrectionType.NONE:
                pass
            case CorrectionType.RAW:
                lai_out = lai * self.is_veg[ts_index]
            case CorrectionType.GSI:
                if self.gsi[ts_index] > self.gsi_threshold:
                    lai_factor = self.correction_factor * self.gsi[ts_index]
                    #lai_factor = self.correction_factor * self.gsi[ts_index]
                    lai_factor = min(lai_factor, 1)
                    lai_out = lai * lai_factor
                    # lai_out = lai * self.gsi[ts_index]
                else:
                    lai_out = 1e-10
            case _:
                raise ValueError(
                    f"Unknown correction type: {self._correction_type}"
                )
        return lai_out

    def compute_photoperiod(
        self,
        date : str,  # date in format 'YYYY/MM/DD HH:MM:SS'
    ):
        # create observer object
        observer = ephem.Observer()
        observer.lat = str(self.latitude)
        observer.long = str(self.longitude)
        observer.date = date

        # create sun object
        sun = ephem.Sun()  # pylint: disable=no-member

        # compute sunrise and sunset
        sunrise = observer.previous_rising(sun)
        sunset = observer.next_setting(sun)

        # compute photoperiod
        photoperiod = sunset - sunrise
        photoperiod = photoperiod * 24

        # return photoperiod
        return photoperiod

    def _build_gsi(self, atm3: plr.DataFrame, use_atm=True) -> None:
        """
        return CANOPY with gsi index
        gsi index is used to rescale maximum LAI
        """
        # from savoy et al 2015
        # modeling the seasonal dynamics of leaf area index
        # # C° Thompson, S.E., et al., 2011.
        # Comparative hydrology across AmeriFlux sites: the variable roles of
        # climate, vegetation, and groundwater. Water Resour. Res. 47 (10)

        # --------------------------------------
        # dataframe subset
        atm = atm3[["TOY", "year", "Tair_C", "vpd", "SW_rad"]]
        atm = atm.with_columns(
            doy = (atm["TOY"] // 1).cast(int)
        )

        # --------------------------------------
        # aggregate values by doy and define:
        # Ta is the minimum daily air temperature
        # Photo is the daily photoperiod
        # VPD is the mean daily vapor pressure deficit VPD.
        atm_day = atm.groupby("doy").agg(
            [
                plr.min("Tair_C"),
                plr.mean("vpd"),
            ]
        ).sort("doy")

        atm_day = atm_day.with_columns(
            photoperiod = atm_day["doy"].apply(
                lambda x: self.compute_photoperiod(x)
            )
        )

        # CONSTANTS
        tmin = -2
        tmax = 20  # °C
        photomin = 10  # 10 hour photoperiod
        photomax = 11  # 11 hour photoperiod
        vpdmax = 4100 / 1000
        vpdmin = 900 / 1000
        if use_atm:
            tmin = atm3["Tair_C"].min()  # type: ignore
            tmax = atm3["Tair_C"].max()  # type: ignore
            vpdmax = atm3["vpd"].max()   # type: ignore
            vpdmin = atm3["vpd"].min()   # type: ignore
            photomin = atm_day["photoperiod"].min()  # type: ignore
            photomax = atm_day["photoperiod"].max()  # type: ignore

        # --------------------------------------
        # !NESTED
        # from Savoy et al 2015 - Modeling the seasonal dynamics of leaf area
        # function to calculate GSI indices (growing season index)
        def get_gsi_index(value, min1, max1, vpd_switch=0):
            if min1 == max1:
                min1 = max1 - 1e12
            match value:
                case _ if value < min1:
                    return 0
                case _ if max1 >= value >= min1:
                    if vpd_switch == 0:
                        return (value - min1) / (max1 - min1)
                    return 1 - ((value - min1) / (max1 - min1))
                case _:
                    return 1

        # --------------------------------------
        # update constants based on the dataset values

        # compute GSI indices
        atm_day = atm_day.with_columns(
            f1_ta = atm_day["Tair_C"].apply(
                lambda x: get_gsi_index(x, tmin, tmax)
            ),
            f2_photo = atm_day["photoperiod"].apply(
                lambda x: get_gsi_index(x, photomin, photomax)
            ),
            f3_vpd = atm_day["vpd"].apply(
                lambda x: get_gsi_index(x, vpdmin, vpdmax, vpd_switch=1)
            )
        )

        # --------------------------------------
        # compute GSI
        atm_day = atm_day.with_columns(
            GSI = atm_day["f1_ta"] * atm_day["f2_photo"] * atm_day["f3_vpd"]
        )

        # smooth GSI using rolling mean ( for dataset with more than 21 days)
        if len(atm_day) > 21:
            # required to avoid NaN values in the first 21 days
            gsi_time_exp = plr.concat(
                [
                    atm_day["GSI"].head(21),
                    atm_day["GSI"],
                ]
            )
            gsi21_time_exp = gsi_time_exp.rolling_mean(21, center=False)
            # remove first 21 values
            atm_day = atm_day.with_columns(
                GSI21 = gsi21_time_exp.tail(len(atm_day))
            )

        # --------------------------------------
        # rescale GSI to 0-1
        atm_day = atm_day.with_columns(
            GSI21s = (
                (atm_day["GSI21"] - atm_day["GSI21"].min())
                / (atm_day["GSI21"].max()     # type: ignore
                   - atm_day["GSI21"].min())  # type: ignore
            )
        )

        # --------------------------------------
        # temporally rescale GSI to input frequency
        daily_gsi = atm_day[["doy", "GSI21s"]]
        atm_df = atm[["doy"]]
        gsi_df = atm_df.join(daily_gsi, on="doy", how="left")
        self.gsi = np.array(gsi_df["GSI21s"])
