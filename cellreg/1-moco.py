import os
import glob
import shutil
from pathlib import Path

from natsort import natsorted
from tqdm import tqdm
import numpy as np

import tifffile
import suite2p

import argparse
import yaml


# this stores constant suite2p ops to force overwrite config
CONSTANT_MOCO_SUITE2P_OPS = {
    'do_registration': 1,
    'reg_tif': True,
    'roidetect': False
}


def get_tiff_list(folder, ext=['.tif', '.tiff']):
    file_list = []
    for x in ext:
        file_list.extend(list(folder.glob(f'*{x}')))
    return file_list

def merge_tiff_stacks(input_tiffs, output_file):
    with tifffile.TiffWriter(output_file) as stack:
        for filename in tqdm(input_tiffs):
            with tifffile.TiffFile(filename) as tif:
                for page in tif.pages:
                    stack.write(
                        page.asarray(), 
                        photometric='minisblack',
                        contiguous=True
                    )
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run motion correction via suite2p (moco) [only 1 channel allow]')

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
    MOCO_CFG_PATH = args.config
    CLEAN_UP = args.cleanup
 
    # define paths
    EXP_DIR = Path(ROOT_DATA_DIR) / SUBJECT_ID / DATE_ID
    RAW_DIR = EXP_DIR / '0-raw' / PLANE_ID
    MOCO_DIR = EXP_DIR / '1-moco' / PLANE_ID
    MOCO_FILEPATH = MOCO_DIR / f'{SUBJECT_ID}-{DATE_ID}-{PLANE_ID}_mc.tif'
    AUX_SUITE2P_DIR = MOCO_DIR / 'suite2p-moco-only'

    # find tiffs
    tiff_list = get_tiff_list(RAW_DIR)
    tiff_list_str = "\n".join(map(str,tiff_list))
    assert len(tiff_list) == 1, \
        f'Currently only accepting one file inside the raw directory "{RAW_DIR}", ' \
        f'found the following: {tiff_list_str}'
        
    raw_tiff_path = tiff_list[0]
    with tifffile.TiffFile(str(raw_tiff_path)) as tif:
        num_frames = len(tif.pages)
        page0 = tif.pages[0]
        Ly, Lx = page0.shape
    print(f'Found raw tiff file ({num_frames} frames of {Ly} x {Lx}): "{raw_tiff_path}"')
    
    # make db    
    db = {
        'data_path': str(MOCO_DIR),
        'save_path0': str(AUX_SUITE2P_DIR),
        'tiff_list': tiff_list,
    }
    print('Data settings: ')
    print(db)
    
    # make ops
    ops = suite2p.default_ops()
    with open(MOCO_CFG_PATH, 'r') as f:
        print(f'Config file loaded from: "{MOCO_CFG_PATH}"')
        ops_cfg = yaml.safe_load(f)
        ops.update(ops_cfg)
        ops.update(CONSTANT_MOCO_SUITE2P_OPS)
        ops.update(db)
        print('Suite2p configuration')
        print(ops)
    
    # convert tiff to binary
    ops = suite2p.io.tiff_to_binary(ops)
    f_raw = suite2p.io.BinaryFile(
        Ly=Ly, Lx=Lx, 
        filename=ops['raw_file']
    )

    # prepare registration file
    f_reg = suite2p.io.BinaryFile(
        Ly=Ly, Lx=Lx, n_frames = f_raw.shape[0],
        filename = ops['reg_file']
    )
    
    # registration
    suite2p.registration_wrapper(
        f_reg, f_raw=f_raw, 
        f_reg_chan2=None, f_raw_chan2=None, align_by_chan2=False,
        refImg=None, ops=ops
    )
    
    # combine registration batches
    REG_TIFF_DIR = Path(ops['reg_file']).parents[0] / 'reg_tif'
    reg_tiff_list = natsorted(list(map(str,REG_TIFF_DIR.glob('*.tif*'))))
    reg_tiff_list_str = "\n".join(reg_tiff_list)
    print(f'Combining the following tiff files into "{MOCO_FILEPATH}":\n{reg_tiff_list_str}')
    
    merge_tiff_stacks(
        input_tiffs=reg_tiff_list,
        output_file=MOCO_FILEPATH
    )
    
    # move files
    if CLEAN_UP:
        SRC_OUT_DIR = os.path.dirname(ops['reg_file'])
        PARENT_SRC_DIR = os.path.dirname(SRC_OUT_DIR)
        assert os.path.basename(PARENT_SRC_DIR) == "suite2p", \
            f'To clean up properly, need "{PARENT_SRC_DIR}" to end with a "suite2p" folder.\n' \
            'Remove "--cleanup" to pass this'
            
        DST_OUT_DIR = str(AUX_SUITE2P_DIR)

        print(f'Moving files from {SRC_OUT_DIR} to {DST_OUT_DIR}')
        shutil.copytree(SRC_OUT_DIR, DST_OUT_DIR, copy_function=shutil.move, dirs_exist_ok=True)
        shutil.rmtree(PARENT_SRC_DIR)

    print(f'Finished motion correction. Use {MOCO_FILEPATH} to continue.')
    
    