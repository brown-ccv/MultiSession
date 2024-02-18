import os
import glob
import shutil
from pathlib import Path
import copy
import multiprocessing as mp
import tempfile

import numpy as np
import pandas as pd
from tqdm import tqdm

import roicat

import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import yaml
import pickle


PARAMS = {
    'um_per_pixel': 0.7,
    'background_max_percentile': 99.9,
    'suite2p': { # `roicat.data_importing.Data_suite2p`
        'new_or_old_suite2p': 'new',
        'type_meanImg': 'meanImg',
    },
    'fov_augment': { # `aligner.augment_FOV_images`
        'roi_FOV_mixing_factor': 0.5,
        'use_CLAHE': False,
        'CLAHE_grid_size': 1,
        'CLAHE_clipLimit': 1,
        'CLAHE_normalize': True,
    },
    'fit_geometric': { # `aligner.fit_geometric`
        'template': 0, 
        'template_method': 'image', 
        'mode_transform': 'affine',
        'mask_borders': (5,5,5,5), 
        'n_iter': 1000,
        'termination_eps': 1e-6, 
        'gaussFiltSize': 15,
        'auto_fix_gaussFilt_step':1,
    },
    'fit_nonrigid': { # `aligner.fit_nonrigid`
        'disable': True,
        'template': 0,
        'template_method': 'image',
        'mode_transform':'createOptFlow_DeepFlow',
        'kwargs_mode_transform':None,
    },
    'roi_blur': {
        'kernel_halfWidth': 2
    }
    
}
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run roicat for multi-session registration')

    parser.add_argument("--root-datadir", type=str)
    parser.add_argument("--subject", type=str)
    parser.add_argument("--plane", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("-g", "--use-gpu", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    parser.add_argument(
        "--max-depth", type=int, default=6,
        help='max depth to find suite2p files, relative to `--root-datadir`'
    )
    
    parser.add_argument(
        "--suite2p-subdir", type=str, default='3-suite2p',
        help='suite2p subdirectory'
    )

    parser.add_argument(
        "--output-topdir", type=str, default='',
        help='output top directory, if empty, will use the subject folder'
    )

    args = parser.parse_args()
    print(args)
    
    # process data paths
    ROOT_DATA_DIR = args.root_datadir
    SUBJECT_ID = args.subject
    PLANE_ID = args.plane
    ROICAT_CFG_PATH = args.config
    SUITE2P_PATH_MAXDEPTH=args.max_depth
    USE_GPU=args.use_gpu
    VERBOSITY = args.verbose
    
    OUTPUT_DIR=args.output_topdir
    SUITE2P_SUBDIR=args.suite2p_subdir
    
    # define paths
    SUBJECT_DIR = Path(ROOT_DATA_DIR) / SUBJECT_ID
    if OUTPUT_DIR in [None, '']:
        OUTPUT_DIR = SUBJECT_DIR
    COLLECTIVE_MUSE_DIR = OUTPUT_DIR / 'multi-session' / PLANE_ID
    COLLECTIVE_MUSE_FIG_DIR = COLLECTIVE_MUSE_DIR / 'figures'
    COLLECTIVE_MUSE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # find suite2p paths
    dir_allOuterFolders = str(SUBJECT_DIR)
    pathSuffixToStat = 'stat.npy'
    pathSuffixToOps = 'ops.npy'
    pathShouldHave = fr'{SUITE2P_SUBDIR}/{PLANE_ID}'
    
    paths_allStat = roicat.helpers.find_paths(
        dir_outer=dir_allOuterFolders,
        reMatch=pathSuffixToStat,
        reMatch_in_path=pathShouldHave,        
        depth=SUITE2P_PATH_MAXDEPTH,
    )[:]

    paths_allStat = [
        x for x in paths_allStat 
        if pathShouldHave in x
    ]
    
    paths_allOps  = np.array([
        Path(path).resolve().parent / pathSuffixToOps
        for path in paths_allStat
    ])[:]


    print('Paths to all suite2p STAT files:')
    print('\n'.join(['\t- ' + str(x) for x in paths_allStat]))
    print('\n')
    print('Paths to all suite2p OPS files:')
    print('\n'.join(['\t- ' + str(x) for x in paths_allOps]))
    print('\n')
    
    # load data    
    data = roicat.data_importing.Data_suite2p(
        paths_statFiles=paths_allStat[:],
        paths_opsFiles=paths_allOps[:],
        um_per_pixel=PARAMS['um_per_pixel'],
        type_meanImg='meanImg', # will be overwritten in the following cell
        **{k: v for k, v in PARAMS['suite2p'].items() if k not in ['type_meanImg']},
        verbose=VERBOSITY,
    )

    assert data.check_completeness(verbose=False)['tracking'],\
        "Data object is missing attributes necessary for tracking."
    
    # also save iscell paths
    data.paths_iscell = [
        Path(x).parent / 'iscell.npy'
        for x in data.paths_ops
    ]
    
    # load all background images
    background_types = [
        'meanImg',
        'meanImgE',
        'max_proj',
        'Vcorr',
    ]

    FOV_backgrounds = {k: [] for k in background_types}
    for ops_file in data.paths_ops:
        ops = np.load(ops_file, allow_pickle=True).item()

        im_sz = (ops['Ly'], ops['Lx'])    
        for bg in background_types:
            bg_im = ops[bg]

            if bg_im.shape == im_sz:
                FOV_backgrounds[bg].append(bg_im)
                continue

            print(
                f'\t- File {ops_file}: {bg} shape is {bg_im.shape}, which is cropped from {im_sz}. '\
                '\n\tWill attempt to add empty pixels to recover the original shape.'
            )

            im = np.zeros(im_sz).astype(bg_im.dtype)
            cropped_xrange, cropped_yrange = ops['xrange'], ops['yrange']
            im[
                cropped_yrange[0]:cropped_yrange[1],
                cropped_xrange[0]:cropped_xrange[1]
            ] = bg_im

            FOV_backgrounds[bg].append(im)

    # choice of FOV images to align
    data.FOV_images = FOV_backgrounds[PARAMS['suite2p']['type_meanImg']]
    
    # obtain FOVs
    aligner = roicat.tracking.alignment.Aligner(verbose=VERBOSITY)

    FOV_images = aligner.augment_FOV_images(
        ims=data.FOV_images,
        spatialFootprints=data.spatialFootprints,
        **PARAMS['fov_augment']
    )
    
    # ALIGN FOV
    DISABLE_NONRIGID = PARAMS['fit_nonrigid'].pop('disable')
    
    # geometric fit
    aligner.fit_geometric(
        ims_moving=FOV_images,
        **PARAMS['fit_geometric']
    )
    aligner.transform_images_geometric(FOV_images)
    remap_idx = aligner.remappingIdx_geo
    
    # non-rigid
    if not DISABLE_NONRIGID:
        aligner.fit_nonrigid(
            ims_moving=aligner.ims_registered_geo,
            remappingIdx_init=aligner.remappingIdx_geo,            
            **PARAMS['fit_nonrigid']
        )
        aligner.transform_images_nonrigid(FOV_images)
        remap_idx = aligner.remappingIdx_nonrigid

    # transform ROIs
    aligner.transform_ROIs(
        ROIs=data.spatialFootprints,
        remappingIdx=remap_idx,
        normalize=True,
    )
    
    # transform other backgrounds
    aligned_backgrounds = {k: [] for k in background_types}
    for bg in background_types:
        aligned_backgrounds[bg] = aligner.transform_images(
            FOV_backgrounds[bg],
            remappingIdx=remap_idx
        )
    
    plt.figure(figsize=(20,20), layout='tight')
    types2plt = background_types + ['ROI']
    nrows = len(types2plt)
    ncols = data.n_sessions
    
    splt_cnt = 1
    for k in types2plt:
        image_list = aligned_backgrounds.get(k, aligner.get_ROIsAligned_maxIntensityProjection())
        for s_id, img in enumerate(image_list):
            plt.subplot(nrows, ncols, splt_cnt)
            plt.imshow(
                img, cmap='Greys_r',
                vmax=np.percentile(
                    img,
                    PARAMS['background_max_percentile'] if k!= "ROI" else 95
                )
            )
            plt.axis('off')
            plt.title(f'Aligned {k} [#{s_id}]') 
            splt_cnt += 1
            
    plt.savefig(COLLECTIVE_MUSE_FIG_DIR / 'aligned-fov.png')
            
    # BUILD FEATUREs
    
    # blur ROI
    blurrer = roicat.tracking.blurring.ROI_Blurrer(
        frame_shape=(data.FOV_height, data.FOV_width),
        plot_kernel=False,
        verbose=VERBOSITY,
        **PARAMS['roi_blur']
    )

    blurrer.blur_ROIs(
        spatialFootprints=aligner.ROIs_aligned[:],
    )
    
    # ROInet embedding
    # TODO: Parameterize `ROInet_embedder`, `generate_dataloader`
    DEVICE = roicat.helpers.set_device(use_GPU=USE_GPU, verbose=VERBOSITY)
    dir_temp = tempfile.gettempdir()

    roinet = roicat.ROInet.ROInet_embedder(
        device=DEVICE,
        dir_networkFiles=dir_temp,
        download_method='check_local_first',
        download_url='https://osf.io/x3fd2/download',
        download_hash='7a5fb8ad94b110037785a46b9463ea94',
        forward_pass_version='latent',
        verbose=VERBOSITY
    )
    
    roinet.generate_dataloader(
        ROI_images=data.ROI_images,
        um_per_pixel=data.um_per_pixel,
        pref_plot=False,
        jit_script_transforms=False,
        batchSize_dataloader=8, 
        pinMemory_dataloader=True,
        numWorkers_dataloader=4,
        persistentWorkers_dataloader=True,
        prefetchFactor_dataloader=2,
    )
    
    roinet.generate_latents()
    
    # Scattering wavelet embedding
    # TODO: Parameterize `SWT`, `SWT.transform`
    swt = roicat.tracking.scatteringWaveletTransformer.SWT(
        kwargs_Scattering2D={'J': 3, 'L': 12}, 
        image_shape=data.ROI_images[0].shape[1:3],
        device=DEVICE,
    )

    swt.transform(
        ROI_images=roinet.ROI_images_rs,
        batch_size=100,
    )
    
    # Compute similarities
    # TODO: Parameterize `ROI_graph`, `compute_similarity_blockwise`, `make_normalized_similarities`
    
    sim = roicat.tracking.similarity_graph.ROI_graph(
        n_workers=-1, 
        frame_height=data.FOV_height,
        frame_width=data.FOV_width,
        block_height=128, 
        block_width=128, 
        algorithm_nearestNeigbors_spatialFootprints='brute',
        verbose=VERBOSITY, 
    )

    s_sf, s_NN, s_SWT, s_sesh = sim.compute_similarity_blockwise(
        spatialFootprints=blurrer.ROIs_blurred,
        features_NN=roinet.latents,
        features_SWT=swt.latents,
        ROI_session_bool=data.session_bool,
        spatialFootprint_maskPower=1.0,
    )
    
    sim.make_normalized_similarities(
        centers_of_mass=data.centroids,
        features_NN=roinet.latents,
        features_SWT=swt.latents, 
        k_max=data.n_sessions*100,
        k_min=data.n_sessions*10,
        algo_NN='kd_tree',
        device=DEVICE,
    )
    
    # Clustering
    # TODO: Parameterize `find_optimal_parameters_for_pruning`?
    clusterer = roicat.tracking.clustering.Clusterer(
        s_sf=sim.s_sf,
        s_NN_z=sim.s_NN_z,
        s_SWT_z=sim.s_SWT_z,
        s_sesh=sim.s_sesh,
    )

    kwargs_makeConjunctiveDistanceMatrix_best = clusterer.find_optimal_parameters_for_pruning(
        n_bins=None, 
        smoothing_window_bins=None,
        kwargs_findParameters={
            'n_patience': 300,
            'tol_frac': 0.001, 
            'max_trials': 1200, 
            'max_duration': 60*10, 
        },
        bounds_findParameters={
            'power_NN': (0., 5.),
            'power_SWT': (0., 5.),
            'p_norm': (-5, 0),
            'sig_NN_kwargs_mu': (0., 1.0), 
            'sig_NN_kwargs_b': (0.00, 1.5), 
            'sig_SWT_kwargs_mu': (0., 1.0),
            'sig_SWT_kwargs_b': (0.00, 1.5),
        },
        n_jobs_findParameters=-1,
    )
    
    kwargs_mcdm_tmp = kwargs_makeConjunctiveDistanceMatrix_best  ## Use the optimized parameters

    clusterer.plot_distSame(kwargs_makeConjunctiveDistanceMatrix=kwargs_mcdm_tmp)
    plt.savefig(COLLECTIVE_MUSE_FIG_DIR / 'pw-sim-distrib.png')

    clusterer.plot_similarity_relationships(
        plots_to_show=[1,2,3], 
        max_samples=100000,  ## Make smaller if it is running too slow
        kwargs_scatter={'s':1, 'alpha':0.2},
        kwargs_makeConjunctiveDistanceMatrix=kwargs_mcdm_tmp
    );
    plt.savefig(COLLECTIVE_MUSE_FIG_DIR / 'pw-sim-scatter.png')
    
    clusterer.make_pruned_similarity_graphs(
        d_cutoff=None,
        kwargs_makeConjunctiveDistanceMatrix=kwargs_mcdm_tmp,
        stringency=1.0,
        convert_to_probability=False,    
    )
    
    if data.n_sessions >= 8:
        labels = clusterer.fit(
            d_conj=clusterer.dConj_pruned,
            session_bool=data.session_bool,
            min_cluster_size=2,
            n_iter_violationCorrection=3,
            split_intraSession_clusters=True,
            cluster_selection_method='leaf',
            d_clusterMerge=None, 
            alpha=0.999, 
            discard_failed_pruning=False, 
            n_steps_clusterSplit=100,
        )

    else:
        labels = clusterer.fit_sequentialHungarian(
            d_conj=clusterer.dConj_pruned,  ## Input distance matrix
            session_bool=data.session_bool,  ## Boolean array of which ROIs belong to which sessions
            thresh_cost=0.6,  ## Threshold 
        )
        
    quality_metrics = clusterer.compute_quality_metrics()
    
    labels_squeezed, labels_bySession, labels_bool, labels_bool_bySession, labels_dict = roicat.tracking.clustering.make_label_variants(labels=labels, n_roi_bySession=data.n_roi)

    results = {
        "clusters":{
            "labels": labels_squeezed,
            "labels_bySession": labels_bySession,
            "labels_bool": labels_bool,
            "labels_bool_bySession": labels_bool_bySession,
            "labels_dict": labels_dict,
        },
        "ROIs": {
            "ROIs_aligned": aligner.ROIs_aligned,
            "ROIs_raw": data.spatialFootprints,
            "frame_height": data.FOV_height,
            "frame_width": data.FOV_width,
            "idx_roi_session": np.where(data.session_bool)[1],
            "n_sessions": data.n_sessions,
        },
        "input_data": {
            "paths_stat": data.paths_stat,
            "paths_ops": data.paths_ops,
        },
        "quality_metrics": clusterer.quality_metrics if hasattr(clusterer, 'quality_metrics') else None,
    }
    
    run_data = copy.deepcopy({
        'data': data.serializable_dict,
        'aligner': aligner.serializable_dict,
        'blurrer': blurrer.serializable_dict,
        'roinet': roinet.serializable_dict,
        'swt': swt.serializable_dict,
        'sim': sim.serializable_dict,
        'clusterer': clusterer.serializable_dict,
    })
    
    iscell_bySession = [np.load(ic_p)[:,0] for ic_p in data.paths_iscell]

    with open(COLLECTIVE_MUSE_DIR / 'roicat-output.pkl', 'wb') as f:
        pickle.dump(dict(
            run_data = run_data,
            results = results,
            iscell = iscell_bySession
        ), f)
        
    print(f'Number of clusters: {len(np.unique(results["clusters"]["labels"]))}')
    print(f'Number of discarded ROIs: {(results["clusters"]["labels"]==-1).sum()}')
    
    # Visualize
    confidence = (((results['quality_metrics']['cluster_silhouette'] + 1) / 2) * results['quality_metrics']['cluster_intra_means'])

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,7))

    axs[0,0].hist(results['quality_metrics']['cluster_silhouette'], 50);
    axs[0,0].set_xlabel('cluster_silhouette');
    axs[0,0].set_ylabel('cluster counts');

    axs[0,1].hist(results['quality_metrics']['cluster_intra_means'], 50);
    axs[0,1].set_xlabel('cluster_intra_means');
    axs[0,1].set_ylabel('cluster counts');

    axs[1,0].hist(confidence, 50);
    axs[1,0].set_xlabel('confidence');
    axs[1,0].set_ylabel('cluster counts');

    axs[1,1].hist(results['quality_metrics']['sample_silhouette'], 50);
    axs[1,1].set_xlabel('sample_silhouette score');
    axs[1,1].set_ylabel('roi sample counts');
    
    fig.savefig(COLLECTIVE_MUSE_FIG_DIR / 'cluster-metrics.png')
    
    # FOV clusters
    FOV_clusters = roicat.visualization.compute_colored_FOV(
        spatialFootprints=[r.power(0.8) for r in results['ROIs']['ROIs_aligned']], 
        FOV_height=results['ROIs']['frame_height'],
        FOV_width=results['ROIs']['frame_width'],
        labels=results["clusters"]["labels_bySession"],  ## cluster labels
        # alphas_labels=confidence*1.5,  ## Set brightness of each cluster based on some 1-D array
        alphas_labels=(clusterer.quality_metrics['cluster_silhouette'] > 0) * (clusterer.quality_metrics['cluster_intra_means'] > 0.4),
    #     alphas_sf=clusterer.quality_metrics['sample_silhouette'],  ## Set brightness of each ROI based on some 1-D array
    )

    FOV_clusters_with_iscell = roicat.visualization.compute_colored_FOV(
        spatialFootprints=[r.power(0.8) for r in results['ROIs']['ROIs_aligned']],  ## Spatial footprint sparse arrays
        FOV_height=results['ROIs']['frame_height'],
        FOV_width=results['ROIs']['frame_width'],
        labels=results["clusters"]["labels_bySession"],  ## cluster labels
        # alphas_labels=confidence*1.5,  ## Set brightness of each cluster based on some 1-D array
        alphas_labels=(clusterer.quality_metrics['cluster_silhouette'] > 0) * (clusterer.quality_metrics['cluster_intra_means'] > 0.4),
        alphas_sf=iscell_bySession
    #     alphas_sf=clusterer.quality_metrics['sample_silhouette'],  ## Set brightness of each ROI based on some 1-D array
    )

    roicat.helpers.save_gif(
        array=FOV_clusters, 
        path=str(COLLECTIVE_MUSE_FIG_DIR/ 'FOV_clusters_allrois.gif'),
        frameRate=5.0,
        loop=0,
    )
    
    roicat.helpers.save_gif(
        array=FOV_clusters_with_iscell, 
        path=str(COLLECTIVE_MUSE_FIG_DIR/ 'FOV_clusters_iscells.gif'),
        frameRate=5.0,
        loop=0,
    )
    
    plt.figure(figsize=(20,10), layout='tight')
    roi_image_dict = {
        'all': FOV_clusters,
        'iscell': FOV_clusters_with_iscell
    }
    nrows = len(roi_image_dict)
    ncols = data.n_sessions
    
    splt_cnt = 1
    for k, image_list in roi_image_dict.items():
        for s_id, img in enumerate(image_list):
            plt.subplot(nrows, ncols, splt_cnt)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Aligned {k} [#{s_id}]') 
            splt_cnt += 1
            
    plt.savefig(COLLECTIVE_MUSE_FIG_DIR / 'aligned-rois.png')
    
    # save FOVs
    num_sessions = data.n_sessions
    out_img = []
    for d in range(num_sessions):
        out_img.append(dict(
            fov = aligner.ims_registered_geo[d],
            roi_pre_iscell = FOV_clusters[d],
            roi_with_iscell = FOV_clusters_with_iscell[d],
            **{k: v[d] for k,v in aligned_backgrounds.items()},
        ))

    with open(COLLECTIVE_MUSE_DIR / 'aligned-images.pkl', 'wb') as f:
        pickle.dump(out_img, f)

    
    # save summary data
    df = pd.DataFrame([
        dict(
            session=i, 
            global_roi=glv, 
            session_roi=range(len(glv)),
            iscell = iscv
        )
        for i, (glv, iscv) in enumerate(zip(labels_bySession, iscell_bySession))
    ]).explode(['global_roi','session_roi', 'iscell']).astype({'iscell': 'bool'})
    
    df.to_csv(COLLECTIVE_MUSE_DIR / 'summary-roi.csv', index=False)
    
    # process only iscell for stats
    df = (
        df.query('iscell')
        .reset_index(drop=True)
    )

    df = df.merge(
        (
            df.query('global_roi >= 0')
            .groupby('global_roi')
            ['session'].agg(lambda x: len(list(x)))
            .to_frame('num_sessions')
            .reset_index()
        ),
        how='left'
    )

    df = (
        df.fillna({'num_sessions': 1})
        .astype({'num_sessions': 'int'})
    )

    # re-indexing
    persistent_roi_reindices = (
        df[['num_sessions', 'global_roi']]
        .query('global_roi >= 0 and num_sessions > 1')
        .drop_duplicates()
        .sort_values('num_sessions', ascending=False)
        .reset_index(drop=True)
        .reset_index()
        .set_index('global_roi')
        ['index'].to_dict()
    )

    df['reindexed_global_roi'] = df['global_roi'].map(persistent_roi_reindices)

    single_roi_start_indices = df['reindexed_global_roi'].max() + 1
    single_roi_rows = df.query('reindexed_global_roi.isna()').index
    num_single_rois = len(single_roi_rows)

    df.loc[single_roi_rows, 'reindexed_global_roi'] = \
        np.arange(num_single_rois) + single_roi_start_indices

    df['reindexed_global_roi'] = df['reindexed_global_roi'].astype('int')
    df = df.rename(columns={
        'global_roi': 'roicat_global_roi', 
        'reindexed_global_roi': 'global_roi'
    })

    df.to_csv(COLLECTIVE_MUSE_DIR / 'finalized-roi.csv', index=False)

    # plot persistent ROIs summary
    persist_rois = (
        df
        .drop_duplicates(['global_roi'])
        .value_counts('num_sessions', sort=False)
        .to_frame('num_rois')
        .reset_index()
    )

    plt.figure(figsize=(4,5))
    ax = sns.barplot(
        persist_rois, 
        x = 'num_sessions',
        y = 'num_rois',
        hue = 'num_sessions',
        facecolor = '#afafaf',
        dodge=False,
        edgecolor='k'
    )
    sns.despine(trim=True, offset=10)

    plt.legend([], [], frameon=False)
    [ax.bar_label(c, padding=5, fontsize=10) for c in ax.containers]
    plt.xlabel('# sessions')
    plt.ylabel('# rois')
    plt.title('Persisted ROIs')
    plt.tight_layout()
    plt.savefig(COLLECTIVE_MUSE_FIG_DIR / 'num-persist-roi-overall.png')

    # plot persistent ROIs per sessions
    df_sessions = (
        df
        .value_counts(['session','num_sessions'])
        .to_frame('count')
        .reset_index()
    )

    df_total_per_session = (
        df_sessions
        .groupby('session')
        ['count'].agg('sum')
        .to_frame('total_count')
        .reset_index()
    )

    df_sessions = df_sessions.merge(df_total_per_session, how='left')
    df_sessions['percent'] = 100 * df_sessions['count'] / df_sessions['total_count']

    plt.figure(figsize=(10,5))
    bar_kwargs = dict(
        kind='bar',
        stacked=True, 
        colormap='GnBu', 
        width=0.7, 
        edgecolor='k',
    )

    ax1 = plt.subplot(121)

    (
        df_sessions
        .pivot(index='session',columns='num_sessions', values='count')
        .fillna(0)
        .plot(
            **bar_kwargs,
            xlabel='session ID',
            ylabel='# rois',
            legend=False,
            ax=ax1)
    )
    plt.tick_params(rotation=0)

    ax2 = plt.subplot(122)

    (
        df_sessions
        .pivot(index='session',columns='num_sessions', values='percent')
        .fillna(0)
        .plot(
            **bar_kwargs,
            xlabel='session ID',
            ylabel='% roi per session',
            ax=ax2
        )
    )
    plt.tick_params(rotation=0)

    leg_handles, leg_labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        reversed(leg_handles),
        reversed(leg_labels),
        loc='upper right', 
        bbox_to_anchor=[1.5,1], 
        title='# sessions',
        edgecolor='k',
    )

    sns.despine(trim=True, offset=10)

    plt.suptitle(
        'Distribution of detected and aligned ROIs across sessions',
    )
    plt.tight_layout(w_pad=5)
    plt.savefig(COLLECTIVE_MUSE_FIG_DIR / 'num-persist-roi-per-session.png')

