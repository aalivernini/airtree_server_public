import h5py
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import os


class DotDict(dict):  # @@2
    """dot.notation access to dictionary attributes (pickable) """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

def f_plot_grp(grp, grp_name, path):
    varn2 = list(grp.keys())
    varn_len = len(varn2)

    date_reference = datetime.datetime(2000,1,1)
    time_reference = (date_reference - datetime.datetime(1970, 1, 1)) / datetime.timedelta(seconds=1)
    #time = (grp['time'] + int(time_reference)).astype('datetime64[s]')
    time = (grp['time'] + int(time_reference)).astype('datetime64[ns]')

    fig = make_subplots(rows=varn_len, cols=1, subplot_titles=varn2,     vertical_spacing = 0.003)
    color2 = ['red', 'blue', 'green', 'orange', 'purple', 'black']

    for cnt, ix in enumerate(varn2):
        if len(grp[ix].shape) > 1:
            for n in range(5):
                fig.append_trace(go.Scatter(
                    x=time,
                    y=grp[ix][:, n],
                    name = 'layer ' + str(n + 1),
                    legendgroup = '1',
                    line = {'color' : color2[n]}
                    ), row=cnt + 1, col=1)
        else:
            fig.append_trace(go.Scatter(
            x=time,
            y=grp[ix],
            name = 'total',
            legendgroup = '2',
            line = {'color' : color2[5]}
            ), row=cnt + 1, col=1)

    fig.update_layout(height=600*varn_len, width=1500, title_text=grp_name,   showlegend=False)
    fig.write_html(path)


def plot(out_dir):
    # define paths
    nc_path = os.path.join(out_dir, 'result.nc')
    os.makedirs(os.path.join(out_dir, 'html'), exist_ok=True)
    struct = ['atm','carb', 'eb', 'fluxes_pollutants', 'fluxes_voc', 'rad', 'res', 'tree']
    html_path = {}
    for x in struct:
        html_path[x] = os.path.join(out_dir, 'html', x+'.html')

    # READ NETCDF
    grp2 = ['atm','carb', 'eb', 'fluxes_pollutants', 'fluxes_voc', 'rad', 'res', 'tree']
    grp3 = dict()
    for k in grp2:
        grp3[k] = xr.open_dataset(nc_path, group=k, decode_coords=True)

    # WRITE GRAPHS TO HTML
    # atm
    f_plot_grp(grp3['atm'], 'meteo & pollutants', html_path['atm'])
    # carbon
    f_plot_grp(grp3['carb'], 'carbon', html_path['carb'])
    # resistances
    f_plot_grp(grp3['res'], 'resistances', html_path['res'])
    # plot radiation
    f_plot_grp(grp3['rad'], 'radiation', html_path['rad'])
    # plot energy balance
    f_plot_grp(grp3['eb'], 'energy balance', html_path['eb'])
    # plot pollutants
    f_plot_grp(grp3['fluxes_pollutants'], 'fluxes_pollutants', html_path['fluxes_pollutants'])
    # plot voc
    f_plot_grp(grp3['fluxes_voc'], 'fluxes_voc', html_path['fluxes_voc'])
    # plot tree
    f_plot_grp(grp3['tree'], 'tree', html_path['tree'])




