import os
import shutil
from pathlib import Path
import numpy as np
import suite2p
import argparse
import yaml
from matplotlib import pyplot as plt

def get_tiff_list(folder, ext=['.tif', '.tiff']):
    file_list = []
    for x in ext:
        file_list.extend(list(folder.glob(f'*{x}')))
    return file_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run suite2p')

    parser.add_argument("--root-datadir", type=str)
    parser.add_argument("--subject", type=str)
    parser.add_argument("--date", type=str)
    parser.add_argument("--plane", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--cleanup", action="store_true")

    args = parser.parse_args()
    print(args)
    
    # process data paths
    ROOT_DATA_DIR = args.root_datadir
    SUBJECT_ID = args.subject
    DATE_ID = args.date
    PLANE_ID = args.plane
    SUITE2P_CFG_PATH = args.config
    CLEAN_UP = args.cleanup
 
    # define paths
   
    EXP_DIR = Path(ROOT_DATA_DIR) / SUBJECT_ID / DATE_ID
    DEEPCAD_DIR = EXP_DIR / '2-deepcad' / PLANE_ID
    SUITE2P_DIR = EXP_DIR / '3-suite2p' / PLANE_ID
    AUX_SUITE2P_DIR = SUITE2P_DIR / 'suite2p-post-deepcad'

    # find tiffs (may be redundant)
    tiff_list = get_tiff_list(DEEPCAD_DIR)

    # make db    
    db = {
        'data_path': str(DEEPCAD_DIR),
        'save_path0': str(AUX_SUITE2P_DIR),
        'tiff_list': tiff_list,
    }
    print('Data settings: ')
    print(db)
    
    # make ops
    ops = suite2p.default_ops()
    with open(SUITE2P_CFG_PATH, 'r') as f:
        print(f'Ops file loaded from: "{SUITE2P_CFG_PATH}"')
        ops_cfg = yaml.safe_load(f)
        ops.update(ops_cfg)
        print('Suite2p configuration')
        print(ops)

    # run suite2p
    output_ops = suite2p.run_s2p(ops=ops, db=db)
    print('Suite2p done!')
    
    # create figure directory
    OUT_DIR = Path(output_ops['save_path'])
    FIG_DIR = OUT_DIR / 'figures'
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # plot and save backgrounds
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.imshow(output_ops['max_proj'], cmap='gray')
    plt.title("Registered Image, Max Projection");
    
    plt.subplot(132)
    plt.imshow(output_ops['meanImg'], cmap='gray')
    plt.title("Mean registered image")
    
    plt.subplot(133)
    plt.imshow(output_ops['meanImgE'], cmap='gray')
    plt.title("High-pass filtered Mean registered image")
    
    fig_bg_file = FIG_DIR / 'backgrounds.png'
    plt.savefig(fig_bg_file, dpi=300)
    print(f'Background images plotted in "{fig_bg_file}"')
    
    # get roi masks
    stats_file = OUT_DIR / 'stat.npy'
    iscell = np.load(OUT_DIR / 'iscell.npy', allow_pickle=True)[:, 0].astype(bool)
    stats = np.load(stats_file, allow_pickle=True)
    
    # plot and save roi masks
    im = suite2p.ROI.stats_dicts_to_3d_array(stats, Ly=output_ops['Ly'], Lx=output_ops['Lx'], label_id=True)
    im[im == 0] = np.nan
    
    plt.figure(figsize=(20,8))
    plt.subplot(1, 4, 1)
    plt.imshow(output_ops['max_proj'], cmap='gray')
    plt.title("Registered Image, Max Projection")
    
    plt.subplot(1, 4, 2)
    plt.imshow(np.nanmax(im, axis=0), cmap='jet')
    plt.title("All ROIs Found")
    
    plt.subplot(1, 4, 3)
    plt.imshow(np.nanmax(im[~iscell], axis=0, ), cmap='jet')
    plt.title("All Non-Cell ROIs")
    
    plt.subplot(1, 4, 4)
    plt.imshow(np.nanmax(im[iscell], axis=0), cmap='jet')
    plt.title("All Cell ROIs")

    fig_roi_file = FIG_DIR / 'roi-masks.png'
    plt.savefig(fig_roi_file, dpi=300)
    print(f'ROI mask plotted in "{fig_roi_file}"')

    # move files
    if CLEAN_UP:
        SRC_OUT_DIR = output_ops['save_path']
        DST_OUT_DIR = str(SUITE2P_DIR)
        print(f'Moving files from {SRC_OUT_DIR} to {DST_OUT_DIR}')
        shutil.copytree(SRC_OUT_DIR, DST_OUT_DIR, copy_function=shutil.move, dirs_exist_ok=True)
        shutil.rmtree(AUX_SUITE2P_DIR)


    print(f'Finished with suite2p. Use {SUITE2P_DIR} to continue.')
