import nibabel as nib
import nrrd
import numpy as np
import os


def compute_laa(img, lung_mask, t=-950):
    """
    Calculate the Low-Attenuation Area Percentage (LAA%) for a given CT scan and lung mask.

    Args:
        img (np.array): The CT scan as a nibabel NIfTI1Image object.
        lung_mask (np.array): The lung mask as a nibabel NIfTI1Image object.
        t (int, optional): The attenuation threshold in Hounsfield Units (default: -950 HU).
    Returns:
        float: The LAA% score, percentage of low-attenuation lung voxels.
    """
    # Ensure that both arrays have the same shape
    if img.shape != lung_mask.shape:
        raise ValueError("CT scan and lung mask must have the same dimensions.")
    total_laa = np.sum(img[lung_mask > 0] < t)
    total_lung = np.sum(lung_mask > 0)
    return (total_laa / total_lung) * 100


def load_patient(path):
    mask_types = ['GTV', 'CTV', 'PTV']
    mask_prefix = ['Lung R+L', 'Lunger H+V']
    ct_path = os.path.join(path, 'planning_ct.nii.gz')

    # sanity checks
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT file not found at {ct_path}")

    mask_path = None
    for prefix in mask_prefix:
        for mask_type in mask_types:
            mask_path = os.path.join(path, f'{prefix} - {mask_type}.nrrd')
            if os.path.exists(mask_path):
                break
        print(f'{prefix} for {path} not available')
    if mask_path is None:
        FileNotFoundError(f'no mask found for {path}')

    ct_scan = nib.load(ct_path)
    ct_data = ct_scan.get_fdata()
    mask_data, _ = nrrd.read(mask_path)

    # Ensure the shapes match
    if ct_data.shape != mask_data.shape:
        raise ValueError(f"CT scan and lung mask dimensions do not match: {ct_data.shape} vs {mask_data.shape}")
    return ct_data, mask_data


def main(root_dir):
    """
    Load CT scans from the given directory structure.

    Args:
        root_dir (str): Root directory containing the CT scans.

    Returns:
        dict: A dictionary where keys are (patient_id, scan_time) tuples and values are nibabel NIfTI1Image objects.
    """
    ct_scans = {}

    # Traverse the directory tree
    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id, '0')
        image, mask = load_patient(patient_path)
        # Check for the planning CT file
        ct_file_path = os.path.join(scan_path, "planning_ct.nii.gz")
        if os.path.exists(ct_file_path):
            try:
                # Load the NIfTI file
                ct_image = nib.load(ct_file_path)
                # Store in the dictionary with (patient_id,