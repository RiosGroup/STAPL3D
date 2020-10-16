
import os
import glob
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import umap
# try:
#     import scanpy as sc
#     sc.settings.verbosity = 2
#     sc.settings.autoshow = False
#     sc.settings.autosave = True
# except:
#     pass
import multiprocessing
# from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from devprof.umap_plotting import plot_umap_series

sc.settings.verbosity = 2
sc.settings.autoshow = False
sc.settings.autosave = True


def read_data(sample_dir, filestem, var_names=[]):

    filename = os.path.join(sample_dir, '{}.csv'.format(filestem))
    adata = sc.read(filename)
    if var_names:
        adata.var_names = var_names

    return adata


def read_data_concat(csv_dir, min_obs=10):

    filenames = os.listdir(csv_dir)
    adatas = []
    for filename in filenames:
        filestem = os.path.splitext(filename)[0]
        filename = os.path.join(csv_dir, '{}.csv'.format(filestem))
        adata = sc.read(filename)
        if adata.shape[0] > min_obs:
            adatas.append(adata)

    adata_all = adata.concatenate(adatas)

    return adata_all


def split_data(adata):

    markers = ['ch{:02d}_mean_intensity'.format(ch) for ch in range(0, 8)]
    marker_data = adata[:, markers]
    marker_data.var_names = [
        'dapi', 'ki67', 'pax8', 'ncam1',
        'six2', 'cadh1', 'cadh6', 'factin'
        ]

    morphs = ['label', 'area', 'centroid-0', 'centroid-1', 'centroid-2']
    morph_data = adata[:, morphs]
    morph_data.var_names = [
        'label', 'volume',
        'com_z', 'com_y', 'com_x'
        ]

    return marker_data, morph_data


def get_var_names(stype='full'):

    var_names = {}

    if stype == 'full':
        var_names['idxs_cols'] = ['idx_seq', 'idx_level_0', 'idx_level_1', 'idx_orig']
    elif stype == 'stack':
        var_names['idxs_cols'] = ['idx_orig']

    var_names['morphs'] = ['label', 'volume', 'com_z', 'com_y', 'com_x']

    var_names['markers'] = ['dapi', 'ki67', 'pax8', 'ncam1', 'six2', 'cadh1', 'cadh6', 'factin']

    var_names['nuclear'] = ['dapi', 'ki67', 'pax8', 'six2']
    var_names['membrane'] = ['ncam1', 'cadh1', 'cadh6', 'factin']

    return var_names


def get_cell_filter(adata, cf,
                    clipping_lower=False, clipping_upper=False,
                    cliprange=[0, 65535]):

    cs = np.ones(adata.n_obs, dtype='bool')

    for k, v in cf.items():

        print('masking cells with {} < {}'.format(k, v[0]))
        cfc = adata.obs[k] > v[0]
        print('     removing # cells: {}'.format(np.sum(~cfc)))
        cs = np.logical_and(cs, cfc)
        print('     remaining # cells: {}'.format(np.sum(cs)))

        print('masking cells with {} > {}'.format(k, v[1]))
        cfc = adata.obs[k] < v[1]
        print('     removing # cells: {}'.format(np.sum(~cfc)))
        cs = np.logical_and(cs, cfc)
        print('     remaining # cells: {}'.format(np.sum(cs)))

    if clipping_lower:
        for i in range(0, adata.X.shape[1]):
            print('masking cells where mean_int of {} is {}'.format(adata.var_names[i], cliprange[0]))
            cond = adata.X[:, i] == cliprange[0]
            print('     removing # cells: {}'.format(np.sum(cond)))
            cs = np.logical_and(cs, ~cond)
            print('     remaining # cells: {}'.format(np.sum(cs)))
    if clipping_upper:
        for i in range(0, adata.X.shape[1]):
            print('masking cells where mean_int of {} is {}'.format(adata.var_names[i], cliprange[1]))
            cond = adata.X[:, i] == cliprange[1]
            print('     removing # cells: {}'.format(np.sum(cond)))
            cs = np.logical_and(cs, ~cond)
            print('     remaining # cells: {}'.format(np.sum(cs)))

    return cs

"""
def get_cell_filter(adata, filters):

    def clipper_filter(feat, filt, cs):

        if 'lower' in filt.keys():
            for f in adata.var_names:
                if f in filt['skipper']:
                    print('skipping {}'.format(f))
                    continue
                print('masking cells where value of {} is {}'.format(f, filt['lower']))
                cond = adata[:, f].X == filt['lower']
                print('     removing # cells: {}'.format(np.sum(cond)))
                cs = np.logical_and(cs, ~cond)
                print('     remaining # cells: {}'.format(np.sum(cs)))

        if 'upper' in filt.keys():
            for f in adata.var_names:
                if f in filt['skipper']:
                    print('skipping {}'.format(f))
                    continue
                print('masking cells where value of {} is {}'.format(f, filt['upper']))
                cond = adata[:, f].X == filt['upper']
                print('     removing # cells: {}'.format(np.sum(cond)))
                cs = np.logical_and(cs, ~cond)
                print('     remaining # cells: {}'.format(np.sum(cs)))

        return cs

    def feature_filter(feat, filt, cs):

        feat_data = adata[:, feat].X

        if 'lower' in filt.keys():
            print('masking cells with {} < {}'.format(feat, filt['lower']))
            cfc = feat_data > filt['lower']
            print('     removing # cells: {}'.format(np.sum(~cfc)))
            cs = np.logical_and(cs, cfc)
            print('     remaining # cells: {}'.format(np.sum(cs)))

        if 'upper' in filt.keys():
            print('masking cells with {} > {}'.format(feat, filt['upper']))
            cfc = feat_data < filt['upper']
            print('     removing # cells: {}'.format(np.sum(~cfc)))
            cs = np.logical_and(cs, cfc)
            print('     remaining # cells: {}'.format(np.sum(cs)))

        return cs

    cs = np.ones([adata.n_obs, 1], dtype='bool')

    for k, filt in filters.items():
        print(k, filt)
        if k == 'clipper':
            cs = clipper_filter(k, filt, cs)
        else:
            cs = feature_filter(k, filt, cs)

    return np.squeeze(cs)
"""

def normalize_adata(adata, method='recipe_MK'):

    if method in ['standard_scaler', 'normalizer', 'minmax', 'power']:
        from sklearn.preprocessing import StandardScaler, Normalizer, minmax_scale, power_transform
        def scaler(x, method):
            if method == 'standard_scaler':
                x_scaled = StandardScaler().fit_transform(x)
            elif method == 'normalizer':
                x_scaled = Normalizer().fit_transform(x)
            elif method == 'minmax':
                x_scaled = minmax_scale(x, feature_range=(0, 1), axis=0, copy=True)
            elif method == 'power':
                x_scaled = power_transform(x, method='yeo-johnson', standardize=True, copy=True)
            return x_scaled
        # TODO: create layer with norm data?
        markers = adata.var_names
        for i, marker in enumerate(markers):
            adata.X[:, i] = scaler(adata.X[:, i].astype(float), method)

    elif method == 'total':
        sc.pp.normalize_total(adata, target_sum=1, key_added='norm_fac')

    elif method == 'scale':
        sc.pp.scale(adata)

    elif method == 'recipe_MK':
        # sc.pp.normalize_total(adata, target_sum=1, key_added='norm_fac')
        sc.pp.log1p(adata)
        sc.pp.scale(adata)

    else:
        pass


def plot_embedding(embedding, c, setname, key, filestem, axranges=[-20, 20, -20, 20]):

    plt.close('all')
    if key == 'pseudotime':
        import matplotlib
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
        plt.scatter(embedding[:, 0], embedding[:, 1], s= 5, c=c, norm=normalize, cmap='Spectral')
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s= 5, c=c, cmap='Spectral')
    plt.xlim(axranges[0], axranges[1])
    plt.ylim(axranges[2], axranges[3])
    plt.title('UMAP embedding of the {} set'.format(setname), fontsize=24);
    plt.savefig('{}_{}_{}.pdf'.format(filestem, setname, key))


def export_trajectory(y, id, key, filestem):

    df = pd.DataFrame(y)
    df.insert(0, 'Id', id)
    df.columns = ['Id', key]
    csvpath = '{}_{}.csv'.format(filestem, key)
    df.to_csv(csvpath, header=True)


def export_for_blocks(y_test, adata, key, output_dir, dm):
    output_dir = os.path.join(output_dir, '{}_csv'.format(key))
    os.makedirs(output_dir, exist_ok=True)
    for k, v in dm.items():
        filestem = os.path.join(output_dir, v)
        mask = adata.obs['idx_level_0'] == k
        export_trajectory(y_test[mask], adata.obs['label'][mask], key, filestem)


def predict_(X_test, trans, key, y_train, classifier, classification, filestem=''):

    # c_trans = y_train.values
    # if classification:
    #     c_trans = c_trans.astype('int')

    clfit = classifier.fit(trans.embedding_, y_train)
    test_embedding = trans.transform(X_test)
    y_test = clfit.predict(test_embedding)
    # c_test = y_test

    # if classification:
    #     c_test = c_test.astype('int')

    # if filestem:
        # plot_embedding(trans.embedding_, c_trans, 'training', key, filestem)
        # plot_embedding(test_embedding, c_test, 'test', key, filestem)
        # export_trajectory(y_test, mdata.X[cs, 0], key, output_dir)

    return y_test


def get_data(sample_dir, filestem=None, cf={'volume': [], 'com_z': []}, norm_method='recipe_MK'):

    csv_dir = os.path.join(sample_dir, 'feature_csv')
    if filestem is not None:
        alldata = read_data(csv_dir, filestem)
    else:
        alldata = read_data_concat(csv_dir)

    adata, mdata = split_data(alldata)

    cs = get_cell_filter(mdata, cf)
    alldata._inplace_subset_obs(cs)
    adata._inplace_subset_obs(cs)
    mdata._inplace_subset_obs(cs)
    normalize_adata(adata, method=norm_method)

    if filestem is not None:
        idxs = slice(0, adata.shape[0], None)
    else:
        idxs = np.random.choice(range(adata.shape[0]), size=10000, replace=False)
    adata = adata[idxs, :]
    mdata = mdata[idxs, :]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    trans = umap.UMAP(n_neighbors=5, random_state=42).fit(adata.X)

    add_cs = True
    if add_cs:
        cs_add = trans.embedding_[:, 0] < 11
        alldata._inplace_subset_obs(cs_add)
        adata._inplace_subset_obs(cs_add)
        mdata._inplace_subset_obs(cs_add)
        if filestem is not None:
            idxs = slice(0, adata.shape[0], None)
        else:
            idxs = np.random.choice(range(adata.shape[0]), size=10000, replace=False)
        adata = adata[idxs, :]
        mdata = mdata[idxs, :]

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        trans = umap.UMAP(n_neighbors=5, random_state=42).fit(adata.X)

    return adata, mdata, trans


def get_gt(adata, output_dir, filestem):

    ### pseudo-groundtruth for training data ### TODO: y_train from wishbone
    ### CLUSTERS
    res = 0.05
    key = 'leiden-{}'.format(res)
    sc.pp.neighbors(adata, n_neighbors=4)
    sc.tl.leiden(adata, resolution=res, key_added=key)
    adata.obs[key].cat.categories = [str(i+1) for i, j in enumerate(adata.obs[key].cat.categories)]
    y_clusters = adata.obs[key]

    if filestem is not None:
        fstem = os.path.join(output_dir, filestem)
        plot_embedding(trans.embedding_, y_clusters.values.astype('int'), 'training', key, fstem)

    ### PSEUDOTIME
    key = 'dpt_pseudotime'
    adata.uns['iroot'] = np.random.choice(np.where(adata[:, ['six2']].X > np.quantile(adata[:, ['six2']].X, .99))[0])
    # sc.pp.neighbors(adata, n_neighbors=4)
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata, n_dcs=5, n_branchings=0, min_group_size=0.01, allow_kendall_tau_shift=True, copy=False)
    y_pseudotime = adata.obs[key]

    if filestem is not None:
        fstem = os.path.join(output_dir, filestem)
        plot_embedding(trans.embedding_, y_pseudotime.values, 'training', key, fstem)

    return y_clusters, y_pseudotime


def merge_clusters(obs, fwmap):
    newcol = np.zeros_like(obs)
    vals = np.array(obs).astype('int')
    newcol = np.array(fwmap)[vals]
    adata.obs['devtraj'] = newcol
    # print(newcol)
    # adata.obs['devtraj'] = pd.Series(newcol, dtype="category")  # FIXME: categorical


def run_embeddings(adata, plot=False, nn=4):

    ### embedding
    if 'X_pca' not in adata.obsm.keys():
        sc.pp.pca(adata, n_comps=3)
        if plot:
            sc.pl.pca_overview(adata)

    if 'X_umap' not in adata.obsm.keys():
        sc.tl.umap(adata, min_dist=0.1)
        if plot:
            sc.pl.umap(adata)

    if 'neighbors' not in adata.uns.keys():
        sc.pp.neighbors(
            adata, n_neighbors=nn,
            n_pcs=None, use_rep=None, knn=True,
            random_state=0, method='umap',
            metric='euclidean', metric_kwds={}, copy=False,
            )

    if 'X_diffmap' not in adata.obsm.keys():
        sc.tl.diffmap(adata)
        if plot:
            sc.pl.diffmap(adata)


def run_clustering(adata, markers, cs, cluster_resolutions=[0.05], plot=False, nn=4):

    cdata = adata.copy()
    cdata._inplace_subset_var(markers)
    cdata._inplace_subset_obs(cs)

    if 'neighbors' not in cdata.uns.keys():
        sc.pp.neighbors(cdata, n_neighbors=nn,  # between 2 and 100
                        n_pcs=None, use_rep=None, knn=True,
                        random_state=0, method='umap',
                        metric='euclidean', metric_kwds={}, copy=False)

    for res in cluster_resolutions:
        key = 'leiden-{:.02f}'.format(res)
        sc.tl.leiden(cdata, resolution=res, key_added=key)
        # key = 'louvain-{:.02f}'.format(res)
        # sc.tl.louvain(cdata, resolution=res, key_added=key)
        cdata.obs[key].cat.categories = [str(i+1) for i, j in enumerate(cdata.obs[key].cat.categories)]
        adata.obs[key] = cdata.obs[key]
        if plot:
            sc.pl.umap(adata, color=key, color_map="nipy_spectral", save='_{}'.format(key))


def pick_iroot(pdata, marker='six2'):

    data = pdata[:, marker].X
    qdata = np.quantile(data, .99)
    pdata.uns['iroot'] = np.random.choice(np.where(data > qdata)[0])


def run_pseudotime(adata, markers, cs, plot=False, use_wishbone=False):

    pdata = adata.copy()
    pdata._inplace_subset_var(markers)
    pdata._inplace_subset_obs(cs)

    if 'iroot' not in adata.uns.keys():
        pick_iroot(pdata)
        adata.uns['iroot'] = int(pdata[pdata.uns['iroot']].obs.index[0])

    if 'neighbors' not in pdata.uns.keys():
        sc.pp.neighbors(pdata, n_neighbors=15,
                        n_pcs=None, use_rep=None, knn=True,
                        random_state=0, method='umap',
                        metric='euclidean', metric_kwds={}, copy=False)

    if 'X_diffmap' not in pdata.obsm.keys():
        sc.tl.diffmap(pdata)
        if plot:
            sc.pl.diffmap(pdata)

    if use_wishbone:
        key = 'wb_pseudotime'
        # cd ~/workspace wishbone
        # python setup.py install
        # conda install scipy=1.2.1
        from wishbone import wb
        df = pdata.to_df()
        df.index = adata.obs.index
        scdata = wb.SCData(df, data_type='masscyt')
        scdata.tsne = pd.DataFrame(pdata.obsm['X_umap'], index=scdata.data.index, columns=['x', 'y'])
        scdata.diffusion_eigenvectors = pd.DataFrame(pdata.obsm['X_diffmap'], index=scdata.data.index)
        scdata.diffusion_eigenvalues = pd.DataFrame(pdata.uns['diffmap_evals'])
        start_cell = pdata[pdata.uns['iroot']].obs.index[0]

        wbdata = wb.Wishbone(scdata)
        wbdata.run_wishbone(
            start_cell,
            branch=True,
            k=15,
            components_list=[1,2,3,4,5],
            num_waypoints=250,
            )

        pdata.obs[key] = pd.DataFrame(wbdata.trajectory.copy())
        pdata.obs['wb_branches'] = pd.DataFrame(wbdata.branch.copy())
        adata.uns['wb_waypoints'] = pd.DataFrame(wbdata.waypoints.copy())

        br = np.zeros_like(adata.X[:, 0], dtype="uint8")
        br[cs] = pdata.obs['wb_branches']
        adata.obs['wb_branches'] = pd.Series(br, dtype="uint8")
        adata.obs['wb_branches'][:] = br.astype('uint8')

    else:
        key = 'dpt_pseudotime'
        sc.tl.dpt(pdata, n_dcs=5, n_branchings=0, min_group_size=0.01, allow_kendall_tau_shift=True, copy=False)

    # subsetted pdata to full adata
    pt = np.zeros_like(adata.X[:, 0], dtype="float32")
    pt[cs] = pdata.obs[key]
    adata.obs[key] = pd.Series(pt, dtype="float32")
    adata.obs[key][:] = pt.astype('float32')

    if plot:
        sc.pl.umap(adata, color=key, color_map="nipy_spectral", save='_{}'.format(key))


def setup_dirs(ddir, sample_id, blocks='blocks_1280', out='tmp'):

    paths = {}
    paths['sample'] = os.path.join(ddir, 'PMCdata', 'Kidney', sample_id)
    paths['blocks'] = os.path.join(paths['sample'], blocks)
    paths['csv'] = os.path.join(paths['blocks'], 'feature_csv')
    paths['output'] = os.path.join(paths['blocks'], out)
    os.makedirs(paths['output'], exist_ok=True)

    paths['filestem'] = os.path.join(paths['output'], '{}'.format(sample_id))

    return paths


def read_and_move(csv_dir, filestem, var_names, subset_var_key='markers'):

    # read the csv
    all_names = []
    all_names += var_names['idxs_cols']
    all_names += var_names['morphs']
    all_names += var_names['markers']
    adata = read_data(csv_dir, filestem, all_names)

    # move the idx and morphologicals to obs
    to_obs_names = []
    to_obs_names += var_names['idxs_cols']
    to_obs_names += var_names['morphs']
    adata.obs = adata[:, to_obs_names].to_df()

    var_subset = var_names[subset_var_key]
    adata._inplace_subset_var(var_subset)

    return adata


def prep_data(csv_dir, filestem, var_names, subsegmented=False):

    fulldata = read_and_move(csv_dir, filestem, var_names)
    if not subsegmented:
        return fulldata

    # replace membranal features
    # NB: this pertains to splitting segments into nuclear/membrane subsegments

    key = 'memb'
    fs = '{}_{}'.format(filestem, key)
    membdata = read_and_move(csv_dir, fs, var_names, key)
    key = 'nucl'
    fs = '{}_{}'.format(filestem, key)
    nucldata = read_and_move(csv_dir, fs, var_names, key)

    # select only cells that have memb-seg (ws2) for now
    # because cannot select along 2 dims
    nucldata._inplace_subset_obs(membdata.obs.index)
    fulldata._inplace_subset_obs(membdata.obs.index)

    repdata = sc.AnnData(np.concatenate((nucldata.X, membdata.X), axis=1))
    repdata.var_names = var_names['nucl'] + var_names['memb']
    repdata.obs = fulldata.obs  # take the idx and morphs from fulldata

    return repdata


def pick_obs(adata, size=0):

    if not size:
        idxs = list(range(adata.shape[0]))
    else:
        idxs = np.random.choice(range(adata.shape[0]), size=size, replace=False)
    # TODO: more informed pick of subset

    return adata[idxs, :], idxs


def nii_to_adata(filepath, filepath_mask='', slices=[]):

    from wmem import Image, LabelImage, MaskImage

    im = Image(filepath, permission='r')
    im.load()
    if slices:
        im.slices = slices + [slice(0, im.dims[3], 1)]
    data = im.slice_dataset()
    im.close()

    if filepath_mask:
        mim = MaskImage(filepath_mask, permission='r')
        mim.load()
        if slices:
            mim.slices = slices
        mask = mim.slice_dataset()
        mim.close()
        m = np.repeat(np.expand_dims(mask, 3), 8, 3)
        X = np.reshape(data[m], [-1, 8])
    else:
        X = np.reshape(data, [-1, 8])
        mask = []

    adata = sc.AnnData(X=X)

    return adata, mask, im


def mask_to_adata(filepath, adata, mask_key='mask', slices=[]):

    mim = MaskImage(filepath, permission='r')
    mim.load()
    if slices:
        mim.slices = slices
    mask = mim.slice_dataset()
    mim.close()

    adata.obs[mask_key] = np.ravel(mask)


def slices_to_adata(dims, slices, adata, mask_key='block'):

    mask = np.zeros(dims)
    mask[slices[0], slices[1], slices[2]] = True

    adata.obs[mask_key] = np.ravel(mask)


def load_train(filestem_train,
               vname_t='trans', vname_c='leiden-0.10', vname_r='dpt_pseudotime',
               ):

    filepath_train = '{}_adata.h5ad'.format(filestem_train)

    refdata = sc.read(filepath_train)

    with open('{}_{}.pickle'.format(filestem_train, vname_t), 'rb') as f:
        trans = pickle.load(f)
    if vname_c:
        with open('{}_{}_{}.pickle'.format(filestem_train, vname_c, 'svc'), 'rb') as f:
            cfit = pickle.load(f)
    else:
        cfit = None
    if vname_r:
        with open('{}_{}_{}.pickle'.format(filestem_train, vname_r, 'svr'), 'rb') as f:
            rfit = pickle.load(f)
    else:
        rfit = None

    return refdata, trans, cfit, rfit


def predict(X, trans, classifier, filestem='', offset=0):

    test_embedding = trans.transform(X)
    test = classifier.predict(test_embedding)

    if filestem:
        np.save('{}_test_{:09d}.npy'.format(filestem, offset), test)
        np.save('{}_test_embedding_{:09d}.npy'.format(filestem, offset), test_embedding)

    return test, test_embedding


def get_markersets(adata):

    markersets = [list(adata.var_names),
                  ['dapi', 'ki67', 'pax8', 'six2', 'ncam1', 'cadh1', 'cadh6', 'factin'],
                  ['ki67', 'pax8', 'six2', 'ncam1', 'cadh1', 'cadh6'],
                  ['pax8', 'six2', 'ncam1', 'cadh1'],
                  ['pax8', 'six2', 'ncam1', 'cadh1', 'cadh6', 'factin',
                   'inertia_tensor_eigvals-0', 'inertia_tensor_eigvals-2',
                   'dist_to_edge', 'extent'],
                  ['pax8', 'six2', 'ncam1', 'cadh1', 'cadh6', 'factin',
                   'major_axis_length', 'minor_axis_length',
                   'dist_to_edge', 'extent', 'fractional_anisotropy'],
                  ['pax8', 'six2', 'ncam1', 'cadh1', 'cadh6', 'factin',
                   'major_axis_length', 'minor_axis_length',
                   'dist_to_edge', 'extent', 'fractional_anisotropy', 'polarity'],
                  ['pax8', 'six2', 'ncam1', 'cadh1', 'cadh6', 'factin',
                   'major_axis_length', 'minor_axis_length',
                   'extent', 'fractional_anisotropy', 'polarity'],
                  ['pax8', 'six2', 'ncam1', 'cadh1', 'cadh6', 'factin'],
                  ['major_axis_length', 'minor_axis_length', 'extent', 'fractional_anisotropy', 'polarity'],
                  ]

    return markersets


def scale(X, axis=0):

    mean = np.mean(X, axis=axis, dtype=np.float64)
    mean_sq = np.multiply(X, X).mean(axis=axis, dtype=np.float64)
    var = mean_sq - mean ** 2
    var *= X.shape[axis] / (X.shape[axis] - 1)
    scale = np.sqrt(var)
    X -= mean
    scale[scale == 0] = 1e-12
    X /= scale

    return X


def get_adata_selected(adata, nm, ms):

    if nm == 'none':
        adata.X = adata.raw.X
    elif nm:
        adata.X = adata.layers[nm]

    if not isinstance(ms, list):
        markersets = get_markersets(adata)
        ms = markersets[ms]
    adata._inplace_subset_var(ms)

    return adata


def run_basic(inputpath, outputstem, nm='none', ms=0, nn=15, nc=2, cluster_resolutions=[0.40, 0.80]):

    adata = sc.read(inputpath)
    adata = get_adata_selected(adata, nm, ms)

    trans = umap.UMAP(n_neighbors=nn, n_components=nc, random_state=42).fit(adata.X)
    with open('{}.pickle'.format(outputstem), 'wb') as f:
        pickle.dump(trans, f)

    adata.obsm['X_umap'] = trans.embedding_

    sc.pp.neighbors(
        adata, n_neighbors=nn,
        n_pcs=None, use_rep=None, knn=True,
        random_state=0, method='umap',
        metric='euclidean', metric_kwds={}, copy=False,
        )

    # basic clustering
    ckeys = ['leiden-{:.02f}'.format(res) for res in cluster_resolutions]
    for res, key in zip(cluster_resolutions, ckeys):
        sc.tl.leiden(adata, resolution=res, key_added=key)
        cats = [str(i+1) for i, j in enumerate(adata.obs[key].cat.categories)]
        adata.obs[key].cat.categories = cats

    # basic pseudotime
    pkeys = ['dpt_pseudotime']
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata, n_dcs=5, n_branchings=0, min_group_size=0.01, allow_kendall_tau_shift=True, copy=False)

    outputpath = '{}.h5ad'.format(outputstem)
    adata.write(outputpath)

    classifier_training(trans, adata, outputstem, ckeys, pkeys)

    plot_umap_series(outputpath, outputstem, nm, ms, nn, nc,
                     ckeys=ckeys, apath=inputpath)
    # FIXME: Pool may stall, even when n_proc = 1


def generate_par_grid(
        inputpath,
        outputstem,
        set_norm_methods=['none', 'scaled', 'logscaled'],
        set_markersets=[0, 1, 2, 3, 4, 5, 6, 7],
        set_n_neighbors=[5, 15, 30, 60],
        set_n_components=[3],
        cluster_resolutions=[0.40, 0.80],
        ):

    arglist = []
    for nm in set_norm_methods:
        for ms in set_markersets:
            for nn in set_n_neighbors:
                for nc in set_n_components:
                    pf = 'nm{}_ms{:02d}_nn{:02d}_nc{:d}'.format(nm, ms, nn, nc)
                    outstem = '{}_{}'.format(outputstem, pf)
                    args = (inputpath, outstem, nm, ms, nn, nc, cluster_resolutions)
                    arglist.append(args)

    return arglist


def aggregate_par_grid(inputpath, outputstem, arglist, split=0):

    if split:
        # TODO
        n = len(arglist) / 3
        argsplits = {'none': arglist[0:n],
                     'scaled': arglist[n:n*2],
                     'logscaled': arglist[n*2:]}
    else:
        argsplits = {'all': arglist}

    for k, arglist in argsplits.items():

        train = sc.read(inputpath)

        for args in arglist:

            fstem = args[1]
            with open('{}.pickle'.format(fstem), 'rb') as f:
                trans = pickle.load(f)
            adata = sc.read('{}.h5ad'.format(fstem))

            pf = 'nm{}_ms{:02d}_nn{:02d}_nc{:d}'.format(args[2], args[3], args[4], args[5])

            train.obsm['X_umap_{}'.format(pf)] = trans.embedding_

            for key in ['leiden-0.40', 'leiden-0.80', 'dpt_pseudotime']:
                train.obs['{}_{}'.format(key, pf)] = adata.obs[key]

        # pick a default embedding [just take the last one]
        train.obsm['X_umap'] = train.obsm['X_umap_{}'.format(pf)]
        train.X = np.copy(train.raw.X)

        # write the aggregated file
        outputpath = '{}_{}.h5ad'.format(outputstem, k)
        train.write(outputpath)


def classifier_training(trans, adata, filestem, ckeys=[], pkeys=[],
                        cluster_key='', clusters_to_pop=[]):

    for key in ckeys:

        svcfit = SVC().fit(trans.embedding_, adata.obs[key])

        filepath = '{}_{}_{}.pickle'.format(filestem, key, 'svc')
        with open(filepath, 'wb') as f:
            pickle.dump(svcfit, f)

    for key in pkeys:

        adata.obs[key][np.isinf(adata.obs[key])] = 1.0  # TODO: generalize

        svrfit = pop_and_train(trans, adata, key, cluster_key, clusters_to_pop)

        filepath = '{}_{}_{}.pickle'.format(filestem, key, 'svr')
        with open(filepath, 'wb') as f:
            pickle.dump(svrfit, f)

    return adata


def classifier_prediction(
        teststem, trainstem, pf, nm, ms,
        start=0, stop=0,
        ckeys=[], pkeys=[],
        cluster_key='', clusters_to_pop=[],
        ):

    # NOTE: adata includes filtered too???
    fdata = sc.read('{}.h5ad'.format(teststem))
    adata = get_adata_selected(fdata.copy(), nm, ms)

    if stop:
        cs = np.zeros((adata.shape[0],), dtype='bool')
        cs[start:stop] = True
        adata._inplace_subset_obs(cs)
        par_pf = '_{:08d}-{:08d}'.format(start, stop)
    else:
        par_pf = ''

    for key in ckeys:

        filepath = '{}_{}_{}_{}.pickle'.format(trainstem, pf, key, 'svc')
        with open(filepath, 'rb') as f:
            svcfit = pickle.load(f)

        svc_test = svcfit.predict(adata.obsm['X_umap'])

        adata.obs[key] = pd.Series(svc_test, dtype="category")
        adata.obs[key][:] = svc_test
        export_trajectory(adata.obs[key], adata.obs['label'], key, '{}_{}{}'.format(teststem, pf, par_pf))

    for key in pkeys:

        #filepath = '{}_{}_{}_{}.pickle'.format(trainstem, pf, key, 'svr')
        filepath = '{}_{}_{}.pickle'.format(trainstem, key, 'svr')
        with open(filepath, 'rb') as f:
            svrfit = pickle.load(f)

        svr_test = pop_and_predict(svrfit, adata, key, cluster_key, clusters_to_pop)

        adata.obs[key] = pd.Series(svr_test, dtype="float32")
        adata.obs[key][:] = svr_test.astype('float32')
        export_trajectory(adata.obs[key], adata.obs['label'], key, '{}_{}{}'.format(teststem, pf, par_pf))


def pop_and_train(trans, adata, key, cluster_key, clusters_to_pop=[]):

    if not clusters_to_pop:
        return SVR().fit(trans.embedding_, adata.obs[key])

    cs = np.ones_like(adata.X[:,0], dtype='bool')
    for cl in clusters_to_pop:
        cs[adata.obs[cluster_key]==cl] = False
    cdata = adata.copy()
    cdata._inplace_subset_obs(cs)

    return SVR().fit(trans.embedding_[cs, :], cdata.obs[key])


def pop_and_predict(svrfit, adata, key, cluster_key, clusters_to_pop=[]):

    if not clusters_to_pop:
        return svrfit.predict(adata.obsm['X_umap'])

    cs = np.ones_like(adata.X[:, 0], dtype='bool')
    for cl in pop_clusters:
        cs[adata.obs[cluster_key]==cl] = False

    svr_test_part = svrfit.predict(adata[cs].obsm['X_umap'])
    svr_test = pd.Series(np.ones(adata.shape[0]), dtype="float32")
    svr_test[cs] = svr_test_part

    return svr_test


def find_slicing(nrows, n_proc=10):

    step = int(np.ceil(nrows / n_proc))
    starts = np.array((range(0, nrows, step)))
    stops = starts + step
    stops[-1] = min(stops[-1], nrows)

    return zip(starts, stops)


def transform_test(teststem, trainstem, pf, nm, ms, start=0, stop=0):

    adata = sc.read('{}.h5ad'.format(teststem))

    with open('{}_{}.pickle'.format(trainstem, pf), 'rb') as f:
        trans = pickle.load(f)

    if stop:
        cs = np.zeros_like(adata.obs['idx_seq'], dtype='bool')
        cs[start:stop] = True
        adata._inplace_subset_obs(cs)
        par_pf = '_{:08d}-{:08d}'.format(start, stop)
    else:
        par_pf = ''

    udata = get_adata_selected(adata.copy(), nm, ms)
    adata.obsm['X_umap'] = trans.transform(udata.X)
    adata_path = '{}_{}{}.h5ad'.format(teststem, pf, par_pf)
    adata.write(adata_path)

    return adata


"""
def transform_test(teststem, trainstem, nm, ms,
                   nn=30, nc=2,
                   start=0, stop=0,
                   ckeys=[], pkeys=[],
                   cluster_key='', clusters_to_pop=[],
                   from_h5ad=False,
                   ):

    adata = sc.read('{}.h5ad'.format(teststem))

    if from_h5ad:
        import umap
        train = sc.read('{}.h5ad'.format(trainstem))
        trans = umap.UMAP(n_neighbors=nn, n_components=nc, random_state=42).fit(train.X)
    else:  # error like https://github.com/lmcinnes/umap/issues/179
        with open('{}.pickle'.format(trainstem), 'rb') as f:
            trans = pickle.load(f)

    if stop:
        cs = np.zeros((adata.shape[0],), dtype='bool')
        cs[start:stop] = True
        adata._inplace_subset_obs(cs)
        pf = '_{:08d}-{:08d}'.format(start, stop)
    else:
        pf = ''

    udata = get_adata_selected(adata.copy(), nm, ms)

    adata.obsm['X_umap'] = trans.transform(udata.X)

    for key in ckeys:

        filepath = '{}_{}_{}.pickle'.format(trainstem, key, 'svc')
        with open(filepath, 'rb') as f:
            svcfit = pickle.load(f)

        svc_test = svcfit.predict(adata.obsm['X_umap'])

        adata.obs[key] = pd.Series(svc_test, dtype="category")
        adata.obs[key][:] = svc_test

    for key in pkeys:

        filepath = '{}_{}_{}.pickle'.format(trainstem, key, 'svr')
        with open(filepath, 'rb') as f:
            svrfit = pickle.load(f)

        svr_test = pop_and_predict(svrfit, adata, key, cluster_key, clusters_to_pop)

        adata.obs[key] = pd.Series(svr_test, dtype="float32")
        adata.obs[key][:] = svr_test.astype('float32')

    adata.write('{}{}.h5ad'.format(teststem, pf))

    return adata
"""

def merge_adata(filestem, pf=''):

    filepaths = [i for i in glob.glob('{}_????????-????????.h5ad'.format(filestem))]
    adata_umap = sc.read(filepaths[0])
    if len(filepaths) > 1:
        for f in filepaths[1:]:
            adata_umap = sc.AnnData.concatenate(adata_umap, sc.read(f))

    adata = sc.read('{}.h5ad'.format(filestem))

    adata.obs = adata_umap.obs
    adata.uns = adata_umap.uns
    adata.obsm = adata_umap.obsm

    # adata.obsm['X_umap'] = adata_umap.obsm['X_umap']
    # adata.obsm['X_umap_{}'.format(pf)] = adata_umap.obsm['X_umap']

    adata.write('{}.h5ad'.format(filestem))

    # for filepath in filepaths:
    #     os.remove(filepath)

    return adata


def merge_csvs(keys, filestem):

    for key in keys:

        outstem = filestem.replace('train', 'test')
        filepaths = [i for i in glob.glob('{}_????????-????????_{}.csv'.format(outstem, key))]
        df = pd.concat([pd.read_csv(f) for f in filepaths])

        outpath = '{}_{}.csv'.format(outstem, key)
        df.to_csv(outpath, index=False, encoding='utf-8-sig')

        for filepath in filepaths:
            os.remove(filepath)
