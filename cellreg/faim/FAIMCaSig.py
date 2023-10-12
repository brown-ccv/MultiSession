"""Functions for performing fully affine-invariant methods to align FOVs."""

import logging
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np
import pandas as pd
from cellpose import models, transforms
from skimage import io, measure
from skimage.segmentation import find_boundaries

from .asift import affine_detect
from .find_obj import filter_matches, init_feature

logging.basicConfig(level=logging.INFO)


def align_images(
    fov_path: str,
    masks_dir_path: Optional[str] = None,
    output_dir_path: Optional[str] = None,
    preprocess: bool = False,
    diameter: Optional[int] = None,
    template_id: int = 0,
    iteration_count: int = 100,
    method: str = "sift",
    cellpose_args: dict = {"gpu": True, "model_type": "cyto2"},
) -> Tuple[List, List, List]:
    """Perform fully affine invariant method on calcium imaging FOV images.

    The function saves the transformation matrices and the registered FOV images
    under the specified output folder or input folder if not specified.

    Args:
        fov_path (str): Path of the folder containing FOV images.
        masks_dir_path (Optional[str], optional): Path of the folder containing ROI
            masks. If the value is None, the function will automatically extract ROI
            masks using Cellpose. If the ROI masks are already obtained, providing the
            path to the folder can save time. Defaults to None.
        output_dir_path (Optional[str], optional): Directory where outputs are written.
            Defaults to None, meaning outputs will be written in the input directory.
        preprocess (bool, optional): Whether to apply contrast adjustment for the
            original FOV images. Defaults to False.
        diameter (Optional[int], optional): Neuron diameter. If None, the diameter will
            be estimated by Cellpose from the original FOV images. Otherwise, the neuron
            will be detected based on the provided diameter value. Defaults to None.
        template_id (int, optional): Choose which FOV image as a template for alignment.
            Defaults to 0.
        iteration_count (int, optional): The number of iterations for the method.
            Defaults to 100.
        method (str, optional): Name of the fully affine invariant method.
            Options include:'akaze', 'sift', 'surf', 'brisk', and 'orb'.
            Defaults to 'sift'.
        cellpose_args (dict): arguments to models.Cellpose()

    Returns:
        Tuple[List, List, List]: Lists of the transformation matrices,
            registered FOV images,and registered ROI masks respectively.
    """
    filenames = get_file_names(fov_path)
    generate_summary(template_id, filenames)
    images = []

    if preprocess:
        images = enhance_image_contrast(filenames)
    else:
        images = [io.imread(file) for file in filenames]

    num_images = len(images)

    if masks_dir_path is None:
        model = models.Cellpose(**cellpose_args)
        channels = [[0, 0] for _ in range(num_images)]

        masks, _, _, _ = model.eval(images, diameter=diameter, channels=channels)
        roi_masks = generate_rois_mask(masks, images)
    else:
        roi_files = get_file_names(masks_dir_path)
        roi_masks = [io.imread(file) for file in roi_files]

    masks_output_dir = Path(output_dir_path or fov_path) / "ROIs_mask"
    masks_output_dir.mkdir(parents=True, exist_ok=True)

    for i, file in enumerate(filenames):
        io.imsave(masks_output_dir / file.name, roi_masks[i])

    template_image = images[template_id]
    template_image = cv.normalize(
        template_image, template_image, 0, 255, cv.NORM_MINMAX
    )
    template_roi = roi_masks[template_id]

    transformation_matrices = []
    registered_images = []
    registered_rois = []

    method = method.lower()
    logging.info(f"A{method.upper()} is running")
    for j, image in enumerate(images):
        if j != template_id:
            logging.info(f"registering {filenames[j].name}")
            reg_image = cv.normalize(image, image, 0, 255, cv.NORM_MINMAX)
            reg_roi = roi_masks[j]
            t_matrix, reg_im, reg_roi = apply_affine_transformation(
                template_image,
                template_roi,
                reg_image,
                reg_roi,
                iteration_count,
                method,
            )
            transformation_matrices.append(t_matrix)
            registered_images.append(reg_im)
            registered_rois.append(reg_roi)

    output_results(
        output_dir_path or fov_path,
        filenames,
        template_id,
        template_image,
        template_roi,
        transformation_matrices,
        registered_images,
        registered_rois,
        method,
    )

    return transformation_matrices, registered_images, registered_rois


def get_file_names(folder: str) -> Optional[List[Path]]:
    """Get image filenames from a folder.

    Args:
        folder (Path): The folder path containing images.

    Returns:
        Union[List[Path], None]: A list of image file paths if found, or None otherwise.

    Raises:
        ValueError: If no images are found or if the folder contains only one image.
    """
    folder = Path(folder)
    image_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    image_paths = [
        image for ext in image_extensions for image in folder.glob(f"*{ext}")
    ]
    unique_image_paths = list(set(image_paths))

    if not unique_image_paths:
        raise ValueError("Load image failed: please check the path")
    elif len(unique_image_paths) == 1:
        raise ValueError("Error: the folder needs to contain at least two images")

    return unique_image_paths


def generate_summary(template_id: int, files: list[str]) -> None:
    """Generates a summary of template and registered images.

    Args:
        template_id (int): The index of the template image in the files list.
        files (list): List of file paths.

    Returns:
        None
    """
    template_image = Path(files[template_id]).name
    logging.info("Template image: %s", template_image)

    regfiles = [Path(files[j]).name for j in range(len(files)) if j != template_id]

    logging.info("Registered images:")
    logging.info(regfiles)


def enhance_image_contrast(image_names: List[str]) -> List[np.ndarray]:
    """Enhance the contrast of a list of images.

    For each image in the list:
    1. Read the image from the provided name.
    2. If the pixel intensity range is more than 0:
        - Normalize it using the `normalize99` function.
        - Clip the pixel intensities to be within the [0, 1] range.
    3. Multiply by 255 and convert to uint8 data type.

    Args:
        image_names: List of image filenames.

    Returns:
        List of enhanced images.
    """
    enhanced_images = []

    for image_name in image_names:
        img = io.imread(image_name)
        if np.ptp(img) > 0:
            img = transforms.normalize99(img)
            img = np.clip(img, 0, 1)
        img *= 255
        img = np.uint8(img)
        enhanced_images.append(img)

    return enhanced_images


def generate_rois_mask(
    masks: List[np.ndarray], imgs: List[np.ndarray]
) -> List[np.ndarray]:
    """Generate a list of masks for ROIs based on input masks andimages.

    Args:
        masks (List[np.ndarray]): A list of masks where each mask corresponds to an
            image in `imgs`. Each pixel in the mask can have a value indicating
            a ROI index.
        imgs (List[np.ndarray]): A list of images.

    Returns:
        List[np.ndarray]: A list of masks for ROIs. For each image, pixels belonging to
            ROIs with more than 60 pixels are set to 255, others are set to 0.
    """
    rois_mask = []

    for mask, img in zip(masks, imgs):
        raw_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for n in range(int(mask.max())):
            ipix = (mask == n + 1).nonzero()
            if len(ipix[0]) > 60:
                raw_mask[ipix[0], ipix[1]] = 255
        rois_mask.append(raw_mask)

    return rois_mask


def apply_affine_transformation(
    target_image: np.ndarray,
    target_image_roi: np.ndarray,
    reference_image: np.ndarray,
    reference_image_roi: np.ndarray,
    iterations: int,
    feature_method: str,
) -> tuple:
    """Apply an affine transformation between a target and reference image.

    Args:
        target_image (np.ndarray): Target image.
        target_image_roi (np.ndarray): Region of Interest (ROI) of the target image.
        reference_image (np.ndarray): Reference image.
        reference_image_roi (np.ndarray): ROI of the reference image.
        iterations (int): Number of iterations for transformation.
        feature_method (str): Method to be used for feature detection.

    Returns:
        tuple: Contains the transformation matrix, warped reference image, and warped
            reference ROI.
    """
    width = target_image.shape[1]
    height = target_image.shape[0]

    feature_name = feature_method + "-flann"
    detector, matcher = init_feature(feature_name)

    # Detect ROI contour on raw ROI
    reference_contours = measure.find_contours(reference_image_roi, 128)
    max_contour_length = max(len(contour) for contour in reference_contours)
    significant_contours_count = sum(
        1
        for contour in reference_contours
        if len(contour) > (max_contour_length * 2 / 3)
    )

    # Detect features
    max_error = width * height * 255
    pool = ThreadPool(processes=cv.getNumberOfCPUs())
    keypoints1, descriptors1 = affine_detect(detector, reference_image, pool=pool)
    keypoints2, descriptors2 = affine_detect(detector, target_image, pool=pool)

    best_H = np.zeros((3, 3))
    best_matches = 0

    for _ in range(iterations):
        temp_roi = deepcopy(target_image_roi)
        raw_matches = matcher.knnMatch(descriptors1, trainDescriptors=descriptors2, k=2)
        p1, p2, kp_pairs = filter_matches(keypoints1, keypoints2, raw_matches)

        if len(p1) >= 4:
            temp_H, status = cv.findHomography(p1, p2, cv.RANSAC, 3.0, 150000)
            kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]

            temp_matches = len(status)
            warped_roi = cv.warpPerspective(
                reference_image_roi, temp_H, (height, width)
            )
            warped_roi[warped_roi < 255] = 0

            # Detect ROI contour on registered ROI
            warped_contours = measure.find_contours(warped_roi, 128)
            warped_significant_contours_count = sum(
                1
                for contour in warped_contours
                if (max_contour_length * 2 / 3) < len(contour) < max_contour_length
            )

            if warped_significant_contours_count < (significant_contours_count / 2):
                continue

            # L1-Norm
            error = np.sum(np.abs(temp_roi - warped_roi))
            if error < max_error:
                max_error = error
                best_H = temp_H
                best_matches = temp_matches
        else:
            best_H, status = None, None
            logging.info(
                "%d matches found, not enough for homography estimation", len(p1)
            )

    if best_matches > 0:
        warped_image = cv.warpPerspective(reference_image, best_H, (height, width))
        warped_image_roi = cv.warpPerspective(
            reference_image_roi, best_H, (height, width)
        )
        transformation_matrix = best_H
    else:
        warped_image = np.zeros([height, width], np.uint8)
        warped_image_roi = np.zeros([height, width], np.uint8)
        transformation_matrix = np.zeros([3, 3])

    return transformation_matrix, warped_image, warped_image_roi


def output_results(
    path: str,
    files: List[Path],
    template_id: int,
    template_image: np.ndarray,
    template_roi: np.ndarray,
    transformation_matrices: List[np.ndarray],
    registered_images: List[np.ndarray],
    registered_rois: List[np.ndarray],
    method: str,
) -> None:
    """Output results including transformation matrices and processed images.

    Args:
        path (Path): Output directory path.
        files (List[Path]): List of file paths.
        template_id (int): ID for the reference/template file in the list of files.
        template_image (np.ndarray): Template image.
        template_roi (np.ndarray): Region of Interest (ROI) of the template image.
        transformation_matrices (List[np.ndarray]): List of transformation matrices.
        registered_images (List[np.ndarray]): List of registered images.
        registered_rois (List[np.ndarray]): List of registered ROIs.
        method (str): Method used to register the images.

    Returns:
        None: Results are saved to disk.
    """
    output_directory = Path(path) / f"A{method.upper()}"
    output_directory.mkdir(parents=True, exist_ok=True)

    k = 0
    for i, file in enumerate(files):
        template_path = Path(files[template_id])
        template_name = template_path.name
        if i != template_id:
            file_path = Path(file)
            raw_data = {
                "Registered_file": [file_path.name],
                "Template_file": [template_name],
                "Transformation_matrix": [transformation_matrices[k]],
            }
            df = pd.DataFrame(raw_data)
            df.to_csv(output_directory / f"{file_path.stem}.csv", index=False)

            output_image = np.zeros(
                [template_image.shape[1], template_image.shape[1], 3], np.uint8
            )
            outlines1 = np.zeros(template_roi.shape, bool)
            outlines1[find_boundaries(template_roi, mode="inner")] = 1
            out_x1, out_y1 = np.nonzero(outlines1)
            output_image[out_x1, out_y1] = [255, 0, 0]

            outlines2 = np.zeros(registered_rois[k].shape, bool)
            outlines2[find_boundaries(registered_rois[k], mode="inner")] = 1
            out_x2, out_y2 = np.nonzero(outlines2)
            output_image[out_x2, out_y2] = [255, 255, 22]

            concatenated_image = cv.hconcat(
                [
                    cv.cvtColor(template_image, cv.COLOR_GRAY2BGR),
                    cv.cvtColor(registered_images[k], cv.COLOR_GRAY2BGR),
                    output_image,
                ]
            )
            io.imsave(output_directory / f"results_{file.name}", concatenated_image)
            k += 1
