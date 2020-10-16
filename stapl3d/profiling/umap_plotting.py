###==========================================================================###
### plotting umaps with plotly: functions
###==========================================================================###
import os
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


def get_orig_df(adata, use_raw=True):

    idx = adata.obs['label'].astype('int')

    if use_raw:
        data = adata.raw.X
    else:
        data = adata.X

    df = pd.DataFrame(data, index=idx, columns=list(adata.var_names))

    return df


def get_umap_df(adata, df, postfix, ckeys=['leiden-0.80'], pkeys=['dpt_pseudotime']):

    idx = df.index

    data = adata.obsm['X_umap{}'.format(postfix)]
    if data.shape[1] > 3:
        data = data[:, :3]
    umap_cols = ['umap-x', 'umap-y', 'umap-z'][:data.shape[1]]
    df2 = pd.DataFrame(data, index=idx, columns=umap_cols)
    df = pd.concat([df, df2], axis=1)

    for ckey in ckeys:
        data = np.array(adata.obs['{}{}'.format(ckey, postfix)].astype('uint16'))
        df2 = pd.DataFrame(data, index=idx, columns=[ckey])
        df = pd.concat([df, df2], axis=1)
        df[ckey] = df[ckey].astype('category')

    for pkey in pkeys:
        data = np.array(adata.obs['{}{}'.format(pkey, postfix)])
        df2 = pd.DataFrame(data, index=idx, columns=[pkey])
        df = pd.concat([df, df2], axis=1)

    # FIXME: get rid of ugly workaround
    if 'leiden-0.80-remap1' in df.columns:
        df['leiden-0.80-remap1'] = df['leiden-0.80-remap1'].astype('category')
        df['dpt_pseudotime'][df['dpt_pseudotime']==0] = 1

    return df


def plot_umap(
    df,
    ckey=None, ccs=None, crange=[0, 0],
    marker={}, camera={}, frames=[],
    outputstem='', fig_formats=[],
    animate=False, picked_set='',
    ):

    if ccs is None:
        ccs = px.colors.sequential.Viridis

    if not camera:
        camera = {
            'up': {'x': 1, 'y': -0.3, 'z': 0},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'eye': {'x': -0.6, 'y': 0, 'z': -1.4},
            # 'projection': 'perspective',
            }

    if picked_set == 'BG':
        bg_vis = True
    else:
        bg_vis = False

    title = dict(
        text=ckey,
        x=0.5, y=0.95,
        xanchor='center',
        yanchor='top',
        )
    axdict = dict(
        visible=bg_vis,
        nticks=5,
        tickfont=dict(color='white', size=1),
        backgroundcolor="rgb(200, 200, 230)",
        gridcolor="white",
        showbackground=True,
        zerolinecolor="white",
        zeroline=True,
        showline=True,
        showgrid=True,
        )
    scene = dict(
        xaxis_title='UMAP-0',
        yaxis_title='UMAP-1',
        xaxis=axdict,
        yaxis=axdict,
        )

    go_3d = 'umap-z' in df.columns

    # some special remapping cases  # TODO: move
    mappings = {
        'leiden-1.50-remap1': (['CM', 'RV/SBB', 'ePT', 'PT', 'DT', 'GL', 'UB', 'CD', 'IP', 'ICc', 'ICm'], css),
        'leiden-1.50-remap2': (['CM', 'RV/SBB', 'ePT', 'PT', 'DT', 'GL', 'UB', 'CD', 'IP', 'ICc', 'ICm'], css),
        'leiden-1.50-remap-NG': (['nonNG', 'CM', 'RV/SBB', 'ePT', 'PT', 'DT', 'GL'], ['#ffffff'] + ccs),
        'leiden-0.40': (['C1', 'C2', 'C3', 'C4', 'C5', 'C6'], css),
        'batch': (['C1', 'C2'], css),
        }
    if ckey == ''

    dfn = df.copy()
    if ckey in mappings.keys():
        cats = mappings[ckey][0]
        css = mappings[ckey][1]
        cdict = {str(i+1): lab for i, lab in enumerate(cats)}
        dfn[ckey].cat.rename_categories(cats, inplace=True)
        labels = {ckey: cats}
        co = {ckey: cats}
    else:
        cats = []
        labels = {}
        co = {}

    if go_3d:
        scene['zaxis_title'] = 'UMAP-2'
        scene['zaxis'] = axdict
        fig = px.scatter_3d(
            dfn, x='umap-x', y='umap-y', z='umap-z',
            color=ckey,
            color_continuous_scale=ccs,
            color_discrete_sequence=ccs,
            category_orders=co,
            # labels=labels,
            range_color=crange,
            )
    else:
        fig = px.scatter(
            dfn, x='umap-x', y='umap-y',
            color=ckey,
            color_continuous_scale=ccs,
            color_discrete_sequence=ccs,
            category_orders=co,
            # labels=labels,
            range_color=crange,
            )

    fig.update_traces(marker=marker, selector=dict(mode='markers'))

    outstem = '{}'.format(outputstem)
    for form in fig_formats:
        if form == 'html':
            fig.update_layout(title=title, scene_camera=camera, scene=scene, showlegend=True)
            size_FG = 2
            size_BG = 4
        elif form == 'pdf':
            if picked_set == 'BG':
                fig.update_layout(title=title, scene_camera=camera, scene=scene, showlegend=True, width=1920, height=1920)
            else:
                fig.update_layout(scene_camera=camera, scene=scene, showlegend=False, width=1920, height=1920)
            fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
            size_FG = 8
            size_BG = 20
        else:
            fig.update_layout(scene_camera=camera, scene=scene, showlegend=False, width=1920, height=1920)
            fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
            size_FG = 8
            size_BG = 20
            # size_FG = 2
            # size_BG = 4

        if ckey == 'leiden-1.50-remap-NG':
            fig.data[0].marker = {'color': '#ffffff', 'symbol': 'circle', 'opacity': 0.1, 'size': 2}
        elif ckey in ['leiden-1.50-remap1', 'leiden-1.50-remap2']:
            sets = {
                'NG': [0, 1, 2, 3, 4, 5],
                'UB': [6, 7],
                'IC': [8, 9, 10],
                'AC': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'NC': [],
                'BG': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                }
            outstem = '{}_{}'.format(outputstem, picked_set)
            for i in range(11):
                if i not in sets[picked_set]:
                    fig.data[i].marker = {'color': '#ffffff', 'symbol': 'circle', 'opacity': 0.1, 'size': size_BG}
                else:
                    fig.data[i].marker = {'color': ccs[i], 'symbol': 'circle', 'opacity': 1.0, 'size': size_FG}
        else:
            if N:
                for i in range(len(cats)):
                    fig.data[i].marker = {'color': ccs[i], 'symbol': 'circle', 'opacity': 1.0, 'size': size_FG}

        if form == 'pngs':
            n_frames = 360
            frames = get_frames_rot360(camera['up'], camera['eye'], n_frames=n_frames)
            create_animation(fig, frames, camera, '{}_frame'.format(outstem))
        else:
            save_fig(fig, outstem, [form])

    return fig


def save_fig(fig, outputstem, fig_formats):
    for fig_format in fig_formats:
        if fig_format == 'html':
            fig.write_html("{}.html".format(outputstem))
        else:
            fig.write_image("{}.{}".format(outputstem, fig_format))


def rotate_z(x, y, z, theta):
    w = x +1j * y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z


def rotate_up(up, eye, theta):
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    k = np.array([up['x'], up['y'], up['z']])
    v = np.array([eye['x'], eye['y'], eye['z']])

    vrot = v*np.cos(theta)+np.cross(k,v)*np.sin(theta)+k*np.dot(k,v)*(1-np.cos(theta))

    return tuple(vrot)


def get_frames_rot360(up, eye, n_frames=48):

    frames=[]
    angles = np.linspace(0, 6.283, n_frames)
    for t in angles:
        # xe, ye, ze = rotate_z(eye['x'], eye['y'], eye['z'], -t)
        xe, ye, ze = rotate_up(up, eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))

    return frames


def get_button():

    updatemenus=[dict(
        type='buttons',
        showactive=False,
        y=1,
        x=1.2,
        xanchor='right',
        yanchor='top',
        pad=dict(t=0, r=10),
        buttons=[dict(label='Play',
                      method='animate',
                      args=[None,
                            dict(frame=dict(duration=50,
                                            redraw=True),
                                 transition=dict(duration=0),
                                 fromcurrent=True,
                                 mode='immediate')
                           ])
                ])
        ]

    return updatemenus


def get_layout_umap3D(ckey='', camera={}, updatemenus=[]):

    title = dict(
        text=ckey,
        x=0.5, y=0.95,
        xanchor='center',
        yanchor='top',
        )
    scene = dict(
        xaxis_title='UMAP-0',
        yaxis_title='UMAP-1',
        zaxis_title='UMAP-2',
        xaxis=dict(
            nticks=5,
            tickfont=dict(color='white', size=1),
            backgroundcolor="rgb(200, 200, 230)",
            gridcolor="white",
            showbackground=False,
            zerolinecolor="white",
            ),
        yaxis=dict(
            nticks=5,
            tickfont=dict(color='white', size=1),
            backgroundcolor="rgb(200, 200, 230)",
            gridcolor="white",
            showbackground=false,
            zerolinecolor="white",
            ),
        zaxis=dict(
            nticks=5,
            tickfont=dict(color='white', size=1),
            backgroundcolor="rgb(200, 200, 230)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white",
            ),
        )
    layout = go.Layout(
        title=title,
        scene_camera=camera,
        scene=scene,
        updatemenus=updatemenus,
        )

    return layout


def get_data_umap3D(df, ckey, cscale='Picnic'):
    x = df['umap-x']
    y = df['umap-y']
    z = df['umap-z']
    if isinstance(df[ckey].dtype, pd.CategoricalDtype):
        c = df[ckey].astype('int')
    else:
        c = df[ckey]
    data = [{
        'hoverlabel': {'namelength': 0},
        'hovertemplate': 'umap-x=%{x}<br>umap-y=%{y}<br>umap-z=%{z}<br>dapi=%{marker.color}',
        'legendgroup': '',
        'marker': {'color': c, 'colorscale': cscale},
        'mode': 'markers',
        'name': '',
        'scene': 'scene',
        'showlegend': False,
        'type': 'scatter3d',
        'x': x, 'y': y, 'z': z}
        ]

    return data


def create_animation(fig, frames, camera, outputstem):

    for i, frame in enumerate(frames):
        eye = frame['layout']['scene']['camera']['eye']
        cam = dict(up=camera['up'], center=camera['center'], eye=eye)
        fig.update_layout(scene_camera=cam)
        outframe = '{}_{:04d}'.format(outputstem, i)
        save_fig(fig, outframe, ['png'])


def get_varnames(colors):

    var_names = {}
    var_names['markers'] = {
        'ncam1': {'crange': [0, 5000], 'color': colors['Ravi']},
        'cadh1': {'crange': [0, 5000], 'color': colors['Ravi']},
        'cadh6': {'crange': [0, 5000], 'color': colors['Ravi']},
        'factin': {'crange': [0, 5000], 'color': colors['Ravi']},
        'dapi': {'crange': [0, 5000], 'color': colors['Ravi']},
        'ki67': {'crange': [0, 5000], 'color': colors['Ravi']},
        'pax8': {'crange': [0, 5000], 'color': colors['Ravi']},
        'six2': {'crange': [0, 5000], 'color': colors['Ravi']},
    }
    var_names['morphs'] = {
        'volume': {'crange': [0, 5000], 'color': colors['Jet']},
        'dist_to_edge': {'crange': [0, 500], 'color': colors['Jet']},
        'equivalent_diameter': {'crange': [0, 20], 'color': colors['Jet']},
        'extent': {'crange': [0, 0.5], 'color': colors['Jet']},
        'major_axis_length': {'crange': [0, 200], 'color': colors['Jet']},
        'minor_axis_length': {'crange': [0, 80], 'color': colors['Jet']},
        'fractional_anisotropy': {'crange': [0.4, 0.8], 'color': colors['Jet']},
        'polarity': {'crange': [0, 0.2], 'color': colors['Jet']},
    }
    # var_names['morphs'] = {
    #     'volume': {'crange': [0, 5000], 'color': colors['Jet']},
    #     'dist_to_edge': {'crange': [0, 500], 'color': colors['Jet']},
    #     'equivalent_diameter': {'crange': [0, 20], 'color': colors['Jet']},
    #     'extent': {'crange': [0, 0.5], 'color': colors['Jet']},
    #     'inertia_tensor_eigvals-0': {'crange': [0, 200], 'color': colors['Jet']},
    #     'inertia_tensor_eigvals-2': {'crange': [0, 200], 'color': colors['Jet']},
    # }
    var_names['clusters'] = {
        'leiden-0.40': {'crange': [0, 0], 'color': None},
        'leiden-0.80': {'crange': [0, 0], 'color': None},
        }
    var_names['vars'] = {
        'dpt_pseudotime': {'crange': [0, 1], 'color': colors['Jet']},
    }

    return var_names


def get_defaults(defaults={}):

    defaults['fig_formats'] = ['pdf', 'html']

    defaults['marker'] = {
        'size': 2, 'opacity': 1.0,
        # 'line': {'width': 1, 'color': 'DarkSlateGrey'},
        }
    defaults['eye'] = {'x': 0.7, 'y': -1.0, 'z': 1.50}
    defaults['center'] = {'x': 0, 'y': 0, 'z': -0.1}
    defaults['camera'] = {
        'up': {'x': 0, 'y': 0, 'z': 1},
        'center': defaults['center'],
        'eye': defaults['eye']
        }

    defaults['frames'] = get_frames_rot360(defaults['camera']['up'], defaults['camera']['eye'], n_frames=96)
    defaults['scene'] = dict(
        xaxis_title='UMAP-0',
        yaxis_title='UMAP-1',
        zaxis_title='UMAP-2',
        xaxis=dict(
            nticks=5,
            tickfont=dict(color='white', size=1),
            backgroundcolor="rgb(200, 200, 230)",
            gridcolor="white",
            showbackground=False,
            zerolinecolor="white",
            ),
        yaxis=dict(
            nticks=5,
            tickfont=dict(color='white', size=1),
            backgroundcolor="rgb(200, 200, 230)",
            gridcolor="white",
            showbackground=False,
            zerolinecolor="white",
            ),
        zaxis=dict(
            nticks=5,
            tickfont=dict(color='white', size=1),
            backgroundcolor="rgb(200, 200, 230)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white",
            ),
        )

    cols = [px.colors.qualitative.Plotly[i]  for i in [1, 5, 7, 2, 8]]  # NG
    cols += [px.colors.qualitative.Set3[i]  for i in [11]]  # GL
    cols += [px.colors.qualitative.Plotly[i]  for i in [0, 3]]  # UB/CD
    cols += [px.colors.qualitative.Plotly[i]  for i in [4]]  # IP
    cols += [px.colors.qualitative.Set2[i]  for i in [-1]]  # ICc
    cols += [px.colors.qualitative.Set1[i]  for i in [-1]]  # ICm
    defaults['colors'] = {
        'Picnic': px.colors.diverging.Picnic,
        'Viridis': px.colors.sequential.Viridis,
        'Rainbow': px.colors.sequential.Rainbow,
        'Jet': px.colors.sequential.Jet,
        # 'JetGrey': px.colors.sequential.Jet + 'rgb(217, 217, 217)',
        'Ravi': ['rgb(217, 217, 217)', 'rgb(0, 0, 131)', 'rgb(250, 0, 0)'],
        'leiden-1.50': cols,
        }

    defaults['var_names'] = get_varnames(defaults['colors'])

    return defaults


def plot_umap_series(
    inputpath, outputstem,
    nm='none', ms=0, nn=15, nc=2,
    mar=None, mor=None,
    ckeys=[], pkeys=[],
    apath=None,
    picked_set='AC',
    fig_formats=['pdf', 'html'],
    ):

    os.makedirs(outputstem, exist_ok=True)

    defaults = get_defaults()
    colors = defaults['colors']

    if mar is None:
        defaults['var_names']['markers'] = {}
    if mor is None:
        defaults['var_names']['morphs'] = {}

    if ckeys:
        defaults['var_names']['clusters'] = {
            k: {'crange': [0, 0], 'color': colors['leiden-1.50']}
            for k in ckeys
            }
    clusters = defaults['var_names']['clusters']
    if pkeys:
        defaults['var_names']['vars'] = {
            k: {'crange': [0, 1], 'color': colors['Jet']}
            for k in pkeys
            }
    pseudotime = defaults['var_names']['vars']

    if apath is not None:
        pf = ''
        adata1 = sc.read(apath)
        adata2 = sc.read(inputpath)
    else:
        pf = 'nm{}_ms{:02d}_nn{:02d}_nc{:d}'.format(nm, ms, nn, nc)
        adata1 = adata2 = sc.read(inputpath)

    df = get_orig_df(adata1, use_raw=True)
    df = get_umap_df(adata2, df, pf, clusters.keys(), pseudotime.keys())

    camera = []  #defaults['camera']

    for group, plot_dict in defaults['var_names'].items():

        for itemname, props in plot_dict.items():

            fig_name = 'umap{}D_{}_{}'.format(nc, group, itemname)
            outstem = os.path.join(outputstem, fig_name)

            fig = plot_umap(
                df, itemname,
                props['color'],
                props['crange'],
                defaults['marker'],
                camera,
                [],
                outstem,
                fig_formats,
                picked_set=picked_set,
                )


def plot_heatmap(df, sort_key, crange, col_names):

    color = 'Viridis'

    dfp = df.sort_values(by=sort_key)

    data=go.Heatmap(
        z=dfp,
        zmin=crange[0], zmax=crange[1],
        colorscale=color,
        x=[vn.upper() for vn in col_names],
        hoverongaps=False,
        type='heatmap',
        )
    fig = go.Figure(data)
    # fig.show()

    fig.update_layout(xaxis_type='category',
                      width=1500, height=900, autosize=False,
                      xaxis={'side': 'top'},
                      xaxis_title='sorted on {}'.format(sort_key),
                      font=dict(family="Helvetica", size=18)
                      )
    fig.update_xaxes(tickfont=dict(color='crimson'), tickangle=-90)
    fig.update_yaxes(autorange="reversed")

    return fig


def plot_matrix(df, cluster_key, crange, col_names):

    dfm = df.groupby(cluster_key).median()
    try:
        dfm = dfm.sort_values(by='dpt_pseudotime')
    except KeyError:
        pass
    dfm = dfm.transpose()

    color='Viridis'
    # col_names = ['dpt_pseudotime']; crange = [0, 1]; color='Jet';
    row_names = dfm.columns
    dfp = dfm.loc[col_names][row_names]

    # fig = px.imshow(dfm[adata.var_names])
    data=go.Heatmap(
        z=dfp,
        zmin=crange[0], zmax=crange[1],
        colorscale=color,
        x=row_names,
        y=[vn.upper() for vn in col_names],
        # xgap=3, ygap=3,
        hoverongaps=False,
        type='heatmap',
        )
    fig = go.Figure(data)
    fig['layout'].update(width=1500, height=900, autosize=False)
    fig.update_layout(xaxis_type='category',
                      yaxis_type='category',
                      width=700, height=900, autosize=False,
                      xaxis={'side': 'top'},
                      xaxis_title=cluster_key,
                      font=dict(family="Helvetica", size=18)
                      )
    fig.update_xaxes(tickfont=dict(color='crimson'), tickangle=-90)
    fig.update_yaxes(autorange="reversed")

    return fig


def plot_correlation_matrix(corr, figpath):

    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.figure.savefig(figpath)
    ax.figure.clf()
