import nibabel as nib
import nrrd
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
import cv2


def extract_mask_outline(mask):
    """
    Extracts the outline of a 3D mask.

    Parameters:
    - mask (np.ndarray): The 3D binary mask array of shape (256, 256, 256).

    Returns:
    - outline_3d (np.ndarray): A 3D array of the same shape as `mask` containing the outlines.
    """
    overlay = np.zeros_like(mask, dtype=np.uint8)
    for i in range(mask.shape[2]):
        mask_slice = mask[:, :, i] * 255
        outline = cv2.Canny(mask_slice, 100, 200)
        overlay[:, :, i] = outline
    return (overlay/255).astype(np.uint8)


def plot_laa_distribution(laa_percentages, save_path = None):
    plt.figure(figsize=(8, 6))
    plt.hist(laa_percentages, bins=20, edgecolor='black', alpha=0.75)
    plt.xlabel('LAA Percentage (%)', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.title('Distribution of LAA Percentages', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if save_path:
        print(f'Save histogram to {save_path}')
        plt.savefig(save_path)


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
    laa = img[lung_mask > 0] < t
    laa_mask = (img*lung_mask < t)*1
    return np.sum(laa) / np.sum(lung_mask > 0)*100, laa_mask.astype(np.uint8)


def categorize_laa(laa_value):
    thresholds = [5, 10, 20]
    categories = ['normal', 'mild', 'moderate', 'severe']
    for i, threshold in enumerate(thresholds):
        if laa_value < threshold:
            return categories[i]
    return categories[-1]


def load_patient(path):
    mask_types = ['GTV', 'CTV', 'PTV']
    mask_prefix = ['Lung R+L', 'Lung R + L', 'Lunger H+V']
    ct_path = os.path.join(path, 'planning_ct.nii.gz')

    # sanity checks
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT file not found at {ct_path}")
    ct_scan = nib.load(ct_path)
    affine = ct_scan.affine
    voxel_sizes = nib.affines.voxel_sizes(affine)
    slice_thickness = voxel_sizes[2]  # Slice thickness is along the z-axis
    print("Voxel sizes (x, y, z):", voxel_sizes)
    ct_data = ct_scan.get_fdata()

    found = False
    for prefix in mask_prefix:
        for mask_type in mask_types:
            mask_path = os.path.join(path, f'{prefix} - {mask_type}.nrrd')
            if os.path.exists(mask_path):
                found = True  # Flag to indicate if a mask was found
                break
        if found:
            break

    if not found:
        return ct_data, (None, None)
    else:
        mask_data, header = nrrd.read(mask_path)

        if ct_data.shape != mask_data.shape:
            raise ValueError(f"CT scan and lung mask dimensions do not match: {ct_data.shape} vs {mask_data.shape}")
        return ct_data, (mask_data, header)


def main(root_dir):
    """
    Load CT scans from the given directory structure.

    Args:
        root_dir (str): Root directory containing the CT scans.

    Returns:
        dict: A dictionary where keys are (patient_id, scan_time) tuples and values are nibabel NIfTI1Image objects.
    """
    laa_counter = Counter({'normal': 0, 'mild': 0, 'moderate': 0, 'severe': 0})

    laas = []
    # Traverse the directory tree
    for patient_id in os.listdir(os.path.join(root_dir, 'nsclc_cbct_dataset')):
        if patient_id.startswith('.'):
            continue
        try:
            int(patient_id)
        except ValueError:
            print(f"Skipping invalid patient ID: {patient_id}")
            continue

        patient_path = os.path.join(root_dir, 'nsclc_cbct_dataset', patient_id, '0')
        image, (mask, mask_header) = load_patient(patient_path)
        if mask is not None:
            laa_perc, laa = compute_laa(image, mask)
            laa_outline = extract_mask_outline(mask)
            laas.append(laa_perc)
            c = categorize_laa(laa_perc)
            laa_counter[c] += 1
            if laa_perc > 5:
                print(f'patient {patient_id} has {c} emphysema!')
                nrrd.write(
                    os.path.join(root_dir, f'maltes_project/emphysema/laa_{c}_p{patient_id}.nrrd'),
                    laa,
                    mask_header
                )

                nrrd.write(
                    os.path.join(root_dir, f'maltes_project/emphysema/laa_outline_{c}_p{patient_id}.nrrd'),
                    laa_outline,
                    mask_header
                )

    # After processing all patients and updating laa_counter
    print("Summary of Emphysema Categories:")
    total_patients = sum(laa_counter.values())

    for category, count in laa_counter.items():
        percentage = (count / total_patients) * 100 if total_patients > 0 else 0
        print(f"{category.capitalize()}: {count} patients ({percentage:.2f}%)")

    print(f"\nTotal Patients Processed: {total_patients}")
    plot_laa_distribution(laas, os.path.join(root_dir, 'maltes_project/emphysema/laa_histogram.png'))


if __name__ == '__main__':
    root = '/media/5tb_encrypted/'
    main(root)
