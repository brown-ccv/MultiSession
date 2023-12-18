"""Performs denoising.

Denoises input images in a given directory with given model and output to an
output directory.
"""

import os
import glob
import shutil
from pathlib import Path

from natsort import natsorted
import tifffile

import argparse
import yaml

from deepcad.test_collection import testing_class

def get_tiff_list(folder, ext=['.tif', '.tiff']):
    file_list = []
    for x in ext:
        file_list.extend(list(folder.glob(f'*{x}')))
    return file_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run denoising with deepcad')

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
    DEEPCAD_CFG_PATH = args.config
    CLEAN_UP = args.cleanup
 
    # define paths
    EXP_DIR = Path(ROOT_DATA_DIR) / SUBJECT_ID / DATE_ID
    MOCO_DIR = EXP_DIR / '1-moco' / PLANE_ID
    DEEPCAD_DIR = EXP_DIR / '2-deepcad' / PLANE_ID
    DEEPCAD_FILEPATH = DEEPCAD_DIR / f'{SUBJECT_ID}-{DATE_ID}-{PLANE_ID}_mc-dc.tif'
    
    # find tiffs
    tiff_list = get_tiff_list(MOCO_DIR)
    tiff_list_str = "\n".join(map(str,tiff_list))
    assert len(tiff_list) == 1, \
        f'Currently only accepting one file inside the motion-corrected directory "{MOCO_DIR}", ' \
        f'found the following: {tiff_list_str}'
        
    tiff_path = tiff_list[0]
    with tifffile.TiffFile(str(tiff_path)) as tif:
        num_frames = len(tif.pages)
        page0 = tif.pages[0]
        Ly, Lx = page0.shape
    print(f'Found motion-corrected tiff file ({num_frames} frames of {Ly} x {Lx}): "{tiff_path}"')
    
    # configuration
    with open(DEEPCAD_CFG_PATH, 'r') as f:
        print(f'Config file loaded from: "{DEEPCAD_CFG_PATH}"')
        dc_cfg = yaml.safe_load(f)
    
    dc_cfg['test_datasize'] = num_frames
    dc_cfg['datasets_path'] = str(MOCO_DIR)
    dc_cfg['output_dir'] = str(DEEPCAD_DIR)
    
    print(dc_cfg)
    
    # deepcad inference
    tc = testing_class(dc_cfg)
    tc.run()

    # get output file
    out_tiff=list(map(str,DEEPCAD_DIR.glob('**/*.tif*')))
    out_tiff=[x for x in out_tiff if os.path.dirname(x) != str(DEEPCAD_DIR)]
    out_tiff_str = "\n".join(map(str,out_tiff))
    assert len(out_tiff) == 1, \
        f'Currently only accepting one output file in deepcad directory for cleaning up "{DEEPCAD_DIR}", ' \
        f'found the following: {out_tiff_str}'
    out_tiff = out_tiff[0]
    
    shutil.copyfile(out_tiff, DEEPCAD_FILEPATH)
    
    # clean up
    if CLEAN_UP:   
        REMOVE_DIR = os.path.dirname(out_tiff)   
        shutil.rmtree(REMOVE_DIR)
    
    print(f'Finished with denoising using deepcad. Use "{DEEPCAD_FILEPATH}" to continue.')