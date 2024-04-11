import numpy as np   # type: ignore
import polars as pl  # type: ignore

# # -- DEBUG MODE--------------------------------------------------------------
#
# """
# DEBUG = 0 # mode=web (minimal output results)
# DEBUG = 1 # mode=disk (minimal output results)
# DEBUG = 2 # mode=disk (all output results)
# """
# DEBUG = 2
#
# import pandas as pd
#
# if DEBUG:
#     from IPython import embed
#     import math
#     import numpy as np
#     import xarray as xr
#     from columnar import columnar
#     import os
#     import pickle
#     import h5py
#     from plotter import plot
#
#
# # -- OPTIONS -----------------------------------------------------------------
#
# ATM_PATH = "../dataset/atm/"
# OUT_DIR = "../dataset/output/"
#
# DEBUG_OPT = {
#     "plot": 1,  # save html plots
#     "save_netcdf": 1,  # save netcdf file
#     "lai_correction": 1,  # apply lai correction for deciduous trees
# }
#
# DEBUG_PARAMETERS = {
#     "param_path": "../dataset/data/param.csv",
#     "atm_path": "../dataset/data/selected_coords.csv",
#     "species_id": 3,
#     "date_start": "01/01/2020",  # format dd/mm/yyyy
#     "date_end": "31/12/2020",  # format dd/mm/yyyy
#     "output_mode": "debug",  # unused, kept for compatibility
#     "output_directory": "/home/web_service_airtree/output",  # unused, kept for compatibility
#     "latitudine": 41.875,
#     "longitudine": 12.625,
#     "altitudine": 130.8,
#     "frequenza": "6",
#     "irrigazione": True,
#     "mode": "disk",  # unused, kept for compatibility
# }
#
# # -- OTHER INFO --------------------------------------------------------------
#
# # MINIMAL INFO REQUIRED TO RUN THE MODEL
# FIELDS = {
#     "CARBON": "c_total,totalRespiration,gpp_ti,carbonStored,npp_m2",
#     "RAD": "fraction_sun_leaves,layer_available_par,par",
#     "RES": (
#         "cuticolar_resistance_O3,atmospheric_resistance,boundary_layer_res_O3,in_canopy_atm_resistance,cuticolar_resistance_NO2,"
#         + "boundary_layer_res_NO2,cuticolar_resistance_SO2,boundary_layer_res_SO2,boundary_layer_res_CO,mean_wind_velocity,USTAR,"
#         + "air_kinematic_viscosity,z_bl,rho_boundary"
#     ),
#     "EB": "tleaf,Gs_leaf_m_s_all,rho_mol",
# }
#
# for key in FIELDS:
#     FIELDS[key] = FIELDS[key].split(",")
#
#
# # -- FUNCTIONS ---------------------------------------------------------------
#

class DotDict(dict):  # @@2
    """dot.notation access to dictionary attributes (pickable)"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


def pr(**kw):
    data2 = []
    for x in kw:
        data = [type(kw[x]), x, kw[x]]
        data2.append(data)
    print(columnar(data2, no_borders=True))


def get_totals_web(FLUXES_POLLUT, CARBON, ATM, frequenza):
    # GET HOURLY VALUES
    timeref = ATM["date"]
    NPP_g_CO2 = CARBON["npp_m2"] * (0.5 * frequenza) / 12 * 44
    O3_gm2 = (-FLUXES_POLLUT["O3flux_tot_sum"] * 1e-9 * 48.0 * 1800) * frequenza
    PM10_gm2 = (FLUXES_POLLUT["pm10_tot_flux"] * 1e-6 * 1800) * frequenza
    PM2_5_gm2 = (FLUXES_POLLUT["pm2_5_tot_flux"] * 1e-6 * 1800) * frequenza
    PM1_gm2 = (FLUXES_POLLUT["pm1_tot_flux"] * 1e-6 * 1800) * frequenza
    NO2_gm2 = (
        -FLUXES_POLLUT["NO2flux_tot_sum"] * 46.0055 * (60 * 30) * 1e-9
    ) * frequenza
    SO2_gm2 = (
        -FLUXES_POLLUT["SO2flux_tot_sum"] * 64.066 * (60 * 30) * 1e-9
    ) * frequenza
    CO_gm2 = (-FLUXES_POLLUT["COflux_tot_sum"] * 28.01 * (60 * 30) * 1e-9) * frequenza

    # GET MONTHLY TOTALS
    data = [
        timeref.to_list(),
        NPP_g_CO2.tolist(),
        O3_gm2.tolist(),
        PM10_gm2.tolist(),
        PM2_5_gm2.tolist(),
        PM1_gm2.tolist(),
        NO2_gm2.tolist(),
        SO2_gm2.tolist(),
        CO_gm2.tolist(),
    ]
    column_names = [
        "date",
        "NPP",
        "O3",
        "PM10",
        "PM2_5",
        "PM1",
        "NO2",
        "SO2",
        "CO",
    ]
    df = pl.from_dict(dict(zip(column_names, data)))
    df = df.with_columns(
            year=df["date"].dt.year(),
            month=df["date"].dt.month(),
            )
    df = df.groupby([df['year'], df['month']]).sum().sort(["year", "month"])

    # avoid null values
    df = df.fill_nan(0)
    print('df null filled', df)

    # create a datetime from year and month columns. Day = 1, hour = 0, minute = 0, second = 0
    df = df.with_columns(
            date=pl.datetime(df["year"], df["month"], 1, 0, 0, 0),
            )

    # convert datetime to timestamp in seconds
    df = df.with_columns(
            time=df["date"].dt.timestamp('ms')// 1000,
            )
    # cleanup
    df = df.drop(["year", "month", "date"])

    # convert dataframe to dictionary. Each key is a column name and each value is a list of values
    tseries = df.to_dict(as_series=False)

    # STORE MONTHLY AND TOTAL RESULTS
    # Results [g m-2] in the time period
    Results                 = DotDict()
    Results['time_series']  = tseries
    Results['total']        = DotDict()
    Results.total["NPP"]    = np.nansum(NPP_g_CO2)
    Results.total["O3"]     = np.nansum(O3_gm2)
    Results.total["PM10"]   = np.nansum(PM10_gm2)
    Results.total["PM2_5"]  = np.nansum(PM2_5_gm2)
    Results.total["PM1"]    = np.nansum(PM1_gm2)
    Results.total["NO2"]    = np.nansum(NO2_gm2)
    Results.total["SO2"]    = np.nansum(SO2_gm2)
    Results.total["CO"]     = np.nansum(CO_gm2)
    return Results
#
#
# def get_totals(FLUXES_POLLUT, CARBON, ATM, CANOPY, frequenza):
#     Results = DotDict()
#     Results["timeref"] = ATM["TIMEs"]
#
#     Results["NPP_tot"] = (
#         np.nansum(CARBON["npp_m2"]) * (0.5 * frequenza) / 12 * 44 / 1000
#     )  # kg CO2 m2 y-1
#     Results["tree_area"] = (
#         (CANOPY["crownwidth1"] / 2) * (CANOPY["crownwidth2"] / 2) * math.pi
#     )  # m2
#     Results["NPP_tot_tree"] = (
#         Results["NPP_tot"] * Results["tree_area"]
#     )  # Kg CO2 tree y-1
#     Results["NPP_hourly"] = CARBON["npp_m2"]  # g(C) h-1 m-2
#
#     Results["o3tot"] = (
#         -np.nansum(FLUXES_POLLUT["O3flux_tot_sum"]) * 1e-9 * 48.0 * 1800
#     ) * frequenza  # g O3 m2 y-1
#     Results["o3tot_tree"] = (Results["o3tot"] * Results["tree_area"])  # g O3 tree y-1
#     Results["o3tot_hourly"] = (
#         -FLUXES_POLLUT["O3flux_tot_sum"] * 1e-6 * 48.0 * 1800
#     )  # mg di O3 m2 h-1
#
#     Results["PM10_flux_tot"] = (
#         np.nansum(FLUXES_POLLUT["pm10_tot_flux"]) * 1e-6 * 1800
#     ) * frequenza  # g PM10 m2 y-1 , moltiplica per 2 se è ogni ora
#     Results["PM10_flux_tree"] = (
#         Results["PM10_flux_tot"] * Results["tree_area"]
#     )  # g di PM10 tree y-1
#     Results["PM10_flux_hourly"] = (
#         FLUXES_POLLUT["pm10_tot_flux"] * 1e-3
#     ) * 1800  # mg m2 h-1
#
#     Results["PM2_5_flux_tot"] = (
#         np.nansum(FLUXES_POLLUT["pm2_5_tot_flux"]) * 1e-6 * 1800
#     ) * frequenza  # g PM2_5 m2 y-1
#     Results["PM2_5_flux_tree"] = (
#         Results["PM2_5_flux_tot"] * Results["tree_area"]
#     )  # g di PM2_5 tree y-1
#     Results["PM2_5_flux_hourly"] = (
#         FLUXES_POLLUT["pm2_5_tot_flux"] * 1e-3
#     ) * 1800  # mg m2 h-1
#
#     Results["PM1_flux_tot"] = (
#         np.nansum(FLUXES_POLLUT["pm1_tot_flux"]) * 1e-6 * 1800
#     ) * frequenza  # g PM1 m2 y-1
#     Results["PM1_flux_tree"] = (
#         Results["PM1_flux_tot"] * Results["tree_area"]
#     )  # g di PM1 tree y-1
#     Results["PM1_flux_hourly"] = (
#         FLUXES_POLLUT["pm1_tot_flux"] * 1e-3
#     ) * 1800  # mg m2 h-1
#
#     Results["NO2_flux_tot"] = (
#         -np.nansum(FLUXES_POLLUT["NO2flux_tot_sum"]) * 46.0055 * (60 * 30) * 1e-9
#     ) * frequenza  # g NO2 m2 yr
#     Results["NO2_flux_tree"] = (
#         Results["NO2_flux_tot"] * Results["tree_area"]
#     )  # g O3 tree y-1
#     Results["NO2_flux_horuly"] = (
#         -FLUXES_POLLUT["NO2flux_tot_sum"] * 1e-6 * 46.0055 * 1800
#     )  # mg di NO2 m2 h-1
#
#     Results["SO2_flux_tot"] = (
#         -np.nansum(FLUXES_POLLUT["SO2flux_tot_sum"]) * 64.066 * (60 * 30) * 1e-9
#     ) * frequenza  # g SO2 m2 yr
#     Results["SO2_flux_tree"] = (
#         Results["SO2_flux_tot"] * Results["tree_area"]
#     )  # g SO2 tree y-1
#     Results["SO2_flux_horuly"] = (
#         -FLUXES_POLLUT["SO2flux_tot_sum"] * 1e-6 * 64.066 * 1800
#     )  # mg di SO2 m2 h-1
#
#     Results["CO_flux_tot"] = (
#         -np.nansum(FLUXES_POLLUT["COflux_tot_sum"]) * 28.01 * (60 * 30) * 1e-9
#     ) * frequenza  # g CO m2 yr
#     Results["CO_flux_tree"] = (
#         Results["CO_flux_tot"] * Results["tree_area"]
#     )  # g CO tree y-1
#     Results["CO_flux_horuly"] = (
#         -FLUXES_POLLUT["COflux_tot_sum"] * 1e-6 * 28.01 * 1800
#     )  # mg di CO m2 h-1
#
#     Results["NPP_kg_CO2"] = Results["NPP_tot"]
#     Results["O3_gm2"] = Results["o3tot"]
#     Results["PM10_gm2"] = Results["PM10_flux_tot"]
#     Results["PM2_5_gm2"] = Results["PM2_5_flux_tot"]
#     Results["NO2_gm2"] = Results["NO2_flux_tot"]
#     Results["SO2_gm2"] = Results["SO2_flux_tot"]
#     Results["CO_gm2"] = Results["CO_flux_tot"]
#     return Results
#
#
# def save_results(
#     Results,
#     canopy_sav,
#     ATM,
#     RAD,
#     RES,
#     EB,
#     CARBON,
#     FLUXES_VOC,
#     FLUXES_POLLUT,
#     TREE,
#     OUT_DIR,
#     time_reference,
# ):
#     os.makedirs(OUT_DIR, exist_ok=True)
#     # --------------------------------------
#     # PRINT TOTALS AND SAVE  THEM TO TEXT FILE
#     n2 = [
#         "NPP_kg_CO2",
#         "O3_gm2",
#         "PM10_gm2",
#         "PM2_5_gm2",
#         "NO2_gm2",
#         "SO2_gm2",
#         "CO_gm2",
#     ]
#     data = []
#     for x in n2:
#         data.append([x, Results[x]])
#     data = columnar(data, headers=["name", "value"], no_borders=True)
#     path = os.path.join(OUT_DIR, "results.txt")
#     with open(path, "w") as f:
#         f.write(data)
#     print(data)
#
#     # --------------------------------------
#     # SAVE CANOPY TO PICKLE
#     path = os.path.join(OUT_DIR, "canopy.pickle")
#     with open(path, "wb") as handle:
#         pickle.dump(canopy_sav, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     # SAVE CANOPY TO TXT FOR EASY READING
#     path = os.path.join(OUT_DIR, "canopy.txt")
#     data = []
#     for x in canopy_sav:
#         data.append([x, canopy_sav[x]])
#     with open(path, "w") as f:
#         data = columnar(data, headers=["name", "value"], no_borders=True)
#         f.write(data)
#
#     # --------------------------------------
#     # SAVE ALL RESULTS TO NETCDF
#     if DEBUG_OPT["save_netcdf"]:
#         out_path = os.path.join(OUT_DIR, "result.nc")
#         if os.path.exists(out_path):
#             os.remove(out_path)
#
#         time = ATM["tempo"].to_numpy()
#         time = time.astype("datetime64[s]") + int(time_reference)
#         time = time.astype("datetime64[ns]")  # TO CHECK
#         arr_x = xr.DataArray(time, dims=["time"])
#         time_len = time.shape[0]
#         arr_layer = xr.DataArray([1, 2, 3, 4, 5], dims=["layer"])
#         data3 = {
#             "rad": RAD,
#             "res": RES,
#             "eb": EB,
#             "carb": CARBON,
#             "fluxes_voc": FLUXES_VOC,
#             "fluxes_pollutants": FLUXES_POLLUT,
#             "tree": TREE,
#             #'result': Results,
#         }
#
#         # COMPRESSION OPTIONS FOR NETCDF
#         comp = dict(zlib=True, complevel=5)
#
#         # STORING AIRTREE RESULTS
#         encoding = {}
#         for ka in data3:
#             ds = xr.Dataset()
#             ds.coords["layer"] = arr_layer
#             ds.coords["time"] = time
#
#             for kb in data3[ka].keys():
#                 try:
#                     encoding[kb] = comp
#                     if kb == "time":
#                         pass
#                         # data = time
#                         # ds.coords['time'] = time
#                     else:
#                         data = data3[ka][kb]
#                         if len(data.shape) == 1:
#                             if data.shape[0] == time_len:
#                                 ds[kb] = (["time"], data)
#                                 # xa = xr.DataArray(data, dims=['time'], coords={'time': arr_x})
#                         else:
#                             if data.shape[1] == 5:
#                                 ds[kb] = (["time", "layer"], data)
#                                 # xa = xr.DataArray(data, dims=['time','layer'], coords={'time': arr_x, 'layer': arr_layer, })
#                 except IndexError as e:
#                     pass
#             ds.to_netcdf(out_path, mode="a", group=ka, engine="h5netcdf")
#
#         # STORING ATM DATA
#         ds = xr.Dataset()
#         encoding = {}
#         ds.coords["time"] = time
#         for x in ATM:
#             encoding[x] = comp
#             data = ATM[x].to_numpy().squeeze()
#             ds[x] = (["time"], data)
#         ds.to_netcdf(out_path, mode="a", group="atm", engine="h5netcdf")
#
#     # --------------------------------------
#     # PLOT RESULTS
#     if DEBUG_OPT["plot"]:
#         plot(OUT_DIR)
