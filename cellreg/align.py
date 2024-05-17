"""Performs FoV alignment.

From a given directory of average images, performs image segmentation and saves output
to a given directory.
"""

from faim.FAIMCaSig import align_images

align_images(
    "/users/yxu150/data/yxu150/projects/FAIMCalcium/plane4_new/",
    preprocess=False,
    diameter=25,
)
