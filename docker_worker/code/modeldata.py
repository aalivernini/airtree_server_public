'''
    convert input data to airtree standard format
'''

import numpy as np
import polars as plr  # type: ignore
from scipy import integrate
from soilcalculator import soilcalculator
from pvlib import location
from pvlib.tools import sind
#from pvlib import tools
#from datetime import datetime



class SolarGeometry:
    def __init__(self, project3, atm3):
        self.project3 = project3
        self.atm3 = atm3


    def insert_solar_geometry(self):
        atm3 = self.atm3.select('TIMEs').with_columns(
            dtime =  plr.from_epoch("TIMEs", time_unit="s")
            )
        dtime = atm3["dtime"]

        # dtime = datetime.utcfromtimestamp(atm3["dtime"])
        loc = location.Location(
            latitude  = self.project3["lat"],
            longitude = self.project3["lon"],
            altitude  = self.project3["altitude"],
        )
        spos = loc.get_solarposition(dtime)
        sun_elevation = spos['elevation']
        sun_elevation[sun_elevation < 0] = 0

        # sun_elevation = np.max(sun_elevation, 0)
        sun_zenith = 90 - sun_elevation

        sinbeta = sind(sun_elevation)
        sinbeta[sinbeta < 0.0001] = 0.0001
        atm3 = self.atm3.with_columns(
                sun_elevation = sun_elevation.to_numpy(),
                sun_zenith = sun_zenith.to_numpy(),
                sinbeta = sinbeta.to_numpy()
        )
        return atm3


class ModelData:
    """Converts input data to airtree standard format.
    Args:
        project3 (project): The project3 parameter.
        canopy3 (dict): The canopy3 parameter.
        atm3 (dataframe): The atm3 parameter.
        atm_data_mode (int): The atm_data_mode parameter.
    """

    def __init__(self, project3, canopy3, atm3, atm_data_mode):
        self.canopy3 = canopy3
        self.atm3 = atm3
        self.atm_data_mode = atm_data_mode
        self.project3 = project3
        super().__init__()

    def define_layer_properties(self):
        '''
        add lai lai fraction,
        layer height
        and emission activity factor to canopy3'''
        canopy3 = self.canopy3

        # compute lai fraction for each canopy layer
        cshape = CrownShape(
            canopy3['depth'],
            canopy3['crownwidth'],
            canopy3['shape']
        )
        ratio = cshape.get_volume_ratio()
        canopy3['layer_volume_ratio'] = ratio

        # compute layer height (midpoint)
        h_base = canopy3["height"] - canopy3["depth"]
        h_step = canopy3["depth"] / canopy3["layers"]
        hcl = [h_base + h_step / 2]
        for ixl in range(1, canopy3['layers']):
            hcl.append(hcl[ixl - 1] + h_step)
        hcl = np.array(hcl)
        canopy3['layer_height'] = hcl[::-1]  # 1st layer is canopy top

    def modeldata(self):
        """get airtree standard format for canopy and atm data"""

        canopy3 = self.canopy3
        canopy3['varname']   = canopy3['species'].strip()
        canopy3["max_lai"] = canopy3.lai
        canopy3["layers"] = int(canopy3["layers"])

        canopy3.lai_coeff = 1
        match (canopy3.transpiretype):
            case 1:  # ipo/epistomatal leaf
                canopy3.lai_coeff = 1
            case 2:  # amphistomatal broadleaves
                canopy3.lai_coeff = 2
            case 3:  # amphistomatal needle-leaf
                canopy3.lai_coeff = 3.14
            case _:
                raise NotImplementedError("transpiretype not implemented")

        # compute soil water content
        model = soilcalculator(
            self.project3.sand ,
            self.project3.silt ,
            self.project3.clay
        )
        [
            canopy3["WP"],
            canopy3["FC"],
            canopy3["SHC"],
            canopy3["BD"],
            canopy3["Porosity"],
        ] = model.soilcalculator()

        if canopy3["depth"] > canopy3["height"]:
            canopy3["depth"] = canopy3["height"]

        match self.atm_data_mode:
            case 1:
                atm3 = self.atm3.rename(
                    {
                        "time": "TIMEs",
                        "air_temperature": "Tair_C",
                        "air_pressure": "P_Pa",
                        "rainfall": "Precip",
                        "shortwave": "SW_rad",
                        "longwave": "LW_rad",
                        "wind_speed": "ws0",
                        "relative_humidity": "RH0",
                        "soil_moisture_10cm": "SWC_10",
                        "soil_moisture_40cm": "SWC_40",
                        "no2": "NO2",
                        "o3": "O3CONC",
                        "co": "CO",
                        "so2": "SO2",
                        "pm10": "pm10",
                        "pm25": "pm2_5",
                        "TIMEd": "TIMEd",
                        "TOY": "TOY",
                    }
                )


                atm3 = atm3.with_columns(
                    USTAR=float("nan"),
                    pm1=atm3["pm10"] / 3.75,
                    CO2conc=390,
                    P_kPa=atm3["P_Pa"] / 1000,
                    Tairk0=atm3["Tair_C"] + 273.15,
                    # da Kg m2 a % vol
                    # kg / m-2
                    SWC_10=(atm3["SWC_10"] / 100) / canopy3["Porosity"],
                    SWC_40=(atm3["SWC_40"] / 300) / canopy3["Porosity"],
                    # weighted average for soil depth
                    theta_avg=(
                        (
                        atm3["SWC_10"] / 100
                        + (atm3["SWC_40"] / 300) * 3)
                        / 4),
                    # theta_avg=(atm3["SWC_40"] / 300) / canopy3["Porosity"],
                )
            case _:
                raise NotImplementedError("ATM_DATA_MODE not implemented")

        # convert posix time to datetime
        atm3 = atm3.with_columns(date=plr.from_epoch("TIMEs", time_unit="s"))

        seconds_from_start = (atm3['date'] - atm3['date'][0]).dt.total_seconds()
        dhh = atm3['date'].dt.hour()
        dmm = atm3['date'].dt.minute()
        dss = atm3['date'].dt.second()
        atm3 = atm3.with_columns(
            TIMEd=seconds_from_start / (24 * 3600),
            TOY=(atm3['date'].dt.ordinal_day() + (
                dhh + dmm / 60 + dss / 3600
            ) / 24),
            year=atm3['date'].dt.year(),
        )  # days from start

        esat_1 = np.exp((17.27 * atm3["Tair_C"]) / (atm3["Tair_C"] + 237.3))
        # saturated vapor pressure (kPa)
        esat = 0.6108 * esat_1

        # eair_kPa: Water vapor pressure (kPa)
        # eair_Pa: Atmospheric vapor pressure (Pa)
        # vpd: VPD in kpa
        atm3 = atm3.with_columns(
            eair_kPa=esat * (atm3["RH0"] / 100),
        )

        atm3 = atm3.with_columns(
            eair_Pa=atm3['eair_kPa'] * 1000,
            vpd=esat - atm3['eair_kPa'],
        )
        self.define_layer_properties()

        # DEV
        solar = SolarGeometry(self.project3, atm3)
        atm3 = solar.insert_solar_geometry()
        return canopy3, atm3


class CrownShape:
    ''' compute the volume ratio of the crown shape'''

    def __init__(self, crown_length: float, crown_wideness: float, shape: str):
        self.crown_wideness = crown_wideness
        self.crown_length = crown_length
        self.shape = shape
        self.smj = crown_length / 2  # hal height
        self.smn = crown_wideness / 2  # crown radius
        self.get_sections()
        match shape:
            case "ellipsoid":
                self.function = self.get_ellipse_y
            case "half-ellipsoid":
                self.function = self.get_half_ellipse_y
            case "conical":
                self.function = self.isosceles_triangle_coordinates
            case "columnar":
                self.function = self.get_ellipse_y  # unused
            case _:
                raise ValueError("Shape not supported")

    def get_sections(self, sections: int = 5):
        smj = self.smj
        # define x threshold for each section
        canopy_step = (smj / sections) * 2
        layer_th2 = [(-smj, -smj + canopy_step)]
        for ixx in range(sections - 1):
            prev = layer_th2[ixx]
            layer_th2.append((prev[1], prev[1] + canopy_step))
        self.threshold = layer_th2

    def get_ellipse_y(self, val_x, smj, smn):
        ''' Compute the y-coordinate for a given x-coordinate on an ellipse
        a: semi-major axis
        b: semi-minor axis
        '''
        value = smn * np.sqrt(1 - (val_x**2 / smj**2))
        return value[value >= 0]

    def get_half_ellipse_y(self, val_x, smj, smn):
        ''' Compute the y-coordinate for a given x-coordinate on an ellipse
        a: semi-major axis
        b: semi-minor axis
        '''
        x = (val_x + smj) / 2
        value = smn * np.sqrt(1 - (x**2 / smj**2))
        return value[value >= 0]

    def isosceles_triangle_coordinates(
        self,
        x_coordinates,
        half_height,
        half_width
    ):
        y_coordinates = (
            half_width
            - (half_width / half_height)
            * x_coordinates
        ) / 2
        return y_coordinates

    def get_volume_ratio(self):
        if self.shape == "cylindrical":
            ones = np.ones(len(self.threshold))
            return ones / len(self.threshold)

        volume2 = []
        for th1 in self.threshold:
            def surface_circle(val_x):
                ''' volume of the curve by integrating the circle slices '''
                radius = self.function(val_x, self.smj, self.smn)
                return np.pi * radius**2
            volume, _ = integrate.quad(surface_circle, th1[0], th1[1])
            volume2.append(volume)
        volume_arr = np.array(volume2)
        volume_tot = np.sum(volume_arr)
        volume_ratio = volume_arr / volume_tot
        return volume_ratio[::-1]  # 1st layer is canopy top


