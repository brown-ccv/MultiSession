"""Performs denoising.

Denoises input images in a given directory with given model and output to an
output directory.
"""

import argparse
import os
from deepcad.test_collection import testing_class

def main():
    parser = argparse.ArgumentParser(description="Performs denoising on input images.")
    
    parser.add_argument("--data_input", type=str, required=True, help="The number of frames to be tested")
    parser.add_argument("--output_directory", type=str, default="./results",
                        help="Output directory for the denoised images")
    parser.add_argument("--denoise_model_path", type=str, required=True, help="Path to the denoise model to use")
    parser.add_argument("--nframes", type=int, default=5640, help="Path to the denoise model to use")


    args = parser.parse_args()

    test_datasize = len([f for f in os.listdir(args.data_input) if os.path.isdir(f)])

    test_dict = {
        'patch_x': 150,
        'patch_y': 150,
        'patch_t': 150,
        'overlap_factor': 0.6,
        'scale_factor': 1,
        'datasets_path': args.data_input,
        'test_datasize': args.nframes,
        'pth_dir': args.denoise_model_path,
        'output_dir': args.output_directory,
        'fmap': 16,
        'GPU': '0',
        'num_workers': 0,
        'visualize_images_per_epoch': False,
        'save_test_images_per_epoch': True
    }

    tc = testing_class(test_dict)
    tc.run()

if __name__ == "__main__":
    main()
