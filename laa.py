import nibabel as nib
import nrrd
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt


def plot_laa_distribution(laa_percentages):
    plt.figure(figsize=(8, 6))
    plt.hist(laa_percentages, bins=10, edgecolor='black', alpha=0.75)
    plt.xlabel('LAA Percentage (%)', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.title('Distribution of LAA Percentages', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return plt

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

    mask_path = None
    for prefix in mask_prefix:
        for mask_type in mask_types:
            mask_path = os.path.join(path, f'{prefix} - {mask_type}.nrrd')
            if os.path.exists(mask_path):
                break
            else:
                mask_path = None
                print(f'{prefix} - {mask_type} for {path} not available')
    if mask_path is None:
        print(os.listdir(path))
        FileNotFoundError(f'no mask found for {path}')

    ct_scan = nib.load(ct_path)
    ct_data = ct_scan.get_fdata()
    mask_data, _ = nrrd.read(mask_path)

    # Ensure the shapes match
    if ct_data.shape != mask_data.shape:
        raise ValueError(f"CT scan and lung mask dimensions do not match: {ct_data.shape} vs {mask_data.shape}")
    print(f'{path} loaded ! wooo' )
    return ct_data, mask_data


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
        image, mask = load_patient(patient_path)
        laa_perc = compute_laa(image, mask)
        print(f'laa perc: {laa_perc}')
        laas.append(laa_perc)
        c = categorize_laa(laa_perc)
        laa_counter[c] += 1

    # After processing all patients and updating laa_counter
    print("Summary of Emphysema Categories:")
    total_patients = sum(laa_counter.values())

    for category, count in laa_counter.items():
        percentage = (count / total_patients) * 100 if total_patients > 0 else 0
        print(f"{category.capitalize()}: {count} patients ({percentage:.2f}%)")

    print(f"\nTotal Patients Processed: {total_patients}")

    plot = plot_laa_distribution(laas)
    plot.savefig(f'{root_dir}/maltes_project/laa_histogram.png')


if __name__ == '__main__':
    root = '/media/5tb_encrypted/'
    main(root)
