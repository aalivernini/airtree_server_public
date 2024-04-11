import math


class soilcalculator:
    def __init__(self, sand, silt, clay):
        self.sand = sand
        self.silt = silt
        self.clay = clay
        super().__init__()

    def soilcalculator(self):
        pctsand = self.sand
        pctclay = self.clay

        acoef_app = (
            -4.396
            - 0.0715 * pctclay
            - 4.88e-4 * pctsand**2
            - 4.285e-5 * pctsand**2 * pctclay
        )
        acoef = math.exp(acoef_app)
        bcoef = -3.14 - 0.00222 * (pctclay**2) - 3.484e-5 * (pctsand**2) * pctclay

        ptcclay_app = math.log10(pctclay)
        sat = 0.332 - 7.251e-4 * pctsand + 0.1276 * ptcclay_app
        acoef_app = 15.0 / acoef
        bcoef_app = 1.0 / bcoef

        wp_raw = pow(acoef_app, bcoef_app)
        fc_raw = (0.333 / acoef) ** (1.0 / bcoef)

        ksat_raw = math.exp(
            (12.012 - 0.0755 * pctsand)
            + (-3.895 + 0.03671 * pctsand - 0.1103 * pctclay + 8.7546e-4 * (pctclay**2))
            / sat
        )
        bd_raw = (1 - sat) * 2.65  # cm3
        #       Corrections from Steve Del Grosso
        WP = wp_raw + (-0.15 * wp_raw)
        FC = fc_raw + (0.07 * fc_raw)
        BD = bd_raw + (-0.08 * bd_raw)  # g/cm3
        # BD = bd_raw
        SHC = ksat_raw / 1500  # saturated hydraulic conductivity mm h-1
        Porosity2 = BD / 2.65
        Porosity = 1 - Porosity2  # cm3 /cm3
        return WP, FC, SHC, BD, Porosity
        # return wp_raw, fc_raw, SHC, BD, Porosity
