"""Performs denoising.

Denoises input images in a given directory with given model and output to an
output directory.
"""

import typer
import os
from deepcad.test_collection import testing_class

app = typer.Typer()

@app.command()
def main(data_input: str = typer.Option(..., "--data-input", help="Path to the directory containing input images to be denoised"),
          output_directory: str = typer.Option("./results", "--output-directory", help="Output directory for the denoised images"),
          denoise_model_path: str = typer.Option(..., "--denoise-model-path", help="Path to the denoise model to use")):
    """
    Performs denoising on input images.
    """
    
    test_datasize = len([f for f in os.listdir(data_input) if os.path.isdir(f)])

    test_dict = {
        'patch_x': 150,
        'patch_y': 150,
        'patch_t': 150,
        'overlap_factor': 0.6,
        'scale_factor': 1,
        'test_datasize': test_datasize,
        'datasets_path': data_input,
        'pth_dir': '.',
        'denoise_model': denoise_model_path,
        'output_dir': output_directory,
        'fmap': 16,
        'GPU': '0',
        'num_workers': 0,
        'visualize_images_per_epoch': False,
        'save_test_images_per_epoch': True
    }

    tc = testing_class(test_dict)
    tc.run()

if __name__ == "__main__":
    app()
