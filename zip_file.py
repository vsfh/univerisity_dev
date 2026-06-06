import zipfile
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import logging
import shutil
from glob import glob
from pathlib import Path

import zipfile
import shutil
from pathlib import Path
import tempfile


def extract_and_copy_400_images():
    # ================== EDIT THESE TWO PATHS ==================
    DEST_DIR = "/home/SATA4T/StereoFromCarla20k"  # CHANGE ME
    # =========================================================

    dest_path = Path(DEST_DIR).expanduser().resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    tmp_dir = "/home/SATA4T/StereoFromCarla_output"
    tmp_path = Path(tmp_dir)

    # Find all 'left' folders to locate triplets
    left_folders = [
        Path(folder) for folder in glob("/home/SATA4T/StereoFromCarla_output/*/*/left")
    ]
    if not left_folders:
        print("No 'left' folders found in zip!")
        return

    target_names = {"left", "right", "depth"}
    processed = set()
    total_copied = 0

    for left_folder in left_folders:
        parent = left_folder.parent
        if parent in processed:
            continue
        processed.add(parent)

        print(f"\nProcessing: {parent.relative_to(tmp_path)}")

        for name in target_names:
            folder = parent / name
            if not folder.is_dir():
                print(f"  [MISSING] {name}/")
                continue

            # Get and sort images
            images = [
                f
                for f in folder.iterdir()
                if f.is_file()
                and f.suffix.lower()
                in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
            ]
            if len(images) < 400:
                print(f"  [SKIP] {name}/ : only {len(images)} images")
                continue

            images.sort(key=lambda x: x.name)
            total = len(images)
            stride = max(1, (total + 399) // 400)
            indices = [i * stride for i in range(400) if i * stride < total]

            # Copy to mirrored dest
            rel = folder.relative_to(tmp_path)
            dest_folder = dest_path / rel
            dest_folder.mkdir(parents=True, exist_ok=True)

            for idx in indices:
                src = images[idx]
                shutil.copy2(src, dest_folder / src.name)

            print(f"  Copied 400 images from {name}/")
            total_copied += 400
        # break
    print(f"\nDone! Copied {total_copied} images total.")
    print(f"Output: {dest_path}")


def zip():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Define paths
    root_dir = Path("/home/SATA4T/StereoFromCarla")
    output_zip = root_dir / "packed_output.zip"
    desired_subfolders = {"left", "depth"}

    def add_folder_to_zip(zip_file, folder_path, relative_path):
        """Add files from a folder to the zip file, preserving relative path."""
        try:
            for item in folder_path.iterdir():
                if item.is_file():
                    zip_file.write(item, relative_path / item.name)
                elif item.is_dir():
                    add_folder_to_zip(zip_file, item, relative_path / item.name)
        except Exception as e:
            logging.error(f"Error processing {folder_path}: {e}")

    def process_folder(folder):
        """Process a single folder and return paths to include in the zip."""
        try:
            folder_paths = []
            for subfolder in folder.iterdir():
                if subfolder.is_dir():
                    # Add left and depth folders
                    for target_folder in desired_subfolders:
                        target_path = subfolder / target_folder
                        if target_path.exists():
                            folder_paths.append(
                                (target_path, target_path.relative_to(root_dir))
                            )
                    # Add right folder from baseline_010
                    baseline_path = subfolder / "baseline_010"
                    right_path = baseline_path / "right"
                    if right_path.exists():
                        relative_path = (subfolder / "right").relative_to(root_dir)
                        folder_paths.append((right_path, relative_path))
            return folder_paths
        except Exception as e:
            logging.error(f"Error processing folder {folder}: {e}")
            return []

    def main():
        # Get list of folders to process
        folders = [f for f in root_dir.iterdir() if f.is_dir()]
        logging.info(f"Found {len(folders)} folders to process.")

        # Use multiprocessing to process folders
        with Pool() as pool:
            results = []
            for folder_paths in tqdm(
                pool.imap_unordered(process_folder, folders),
                total=len(folders),
                desc="Processing folders",
            ):
                results.extend(folder_paths)

        logging.info(f"Total paths to zip: {len(results)}")

        # Create zip file
        with zipfile.ZipFile(
            output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9
        ) as zipf:
            for folder_path, relative_path in tqdm(results, desc="Zipping files"):
                add_folder_to_zip(zipf, folder_path, relative_path)

        logging.info(f"Zip file created at: {output_zip}")


def write_train_test_split_dense_uav():
    from natsort import natsorted
    from glob import glob

    root = "/data/feihong/DenseUAV"
    drone_list = natsorted(glob(f"{root}/*/*rone/*5/H80.JPG"))

    output_path = "/data/feihong/ckpt/test_dense.txt"
    with open(output_path, "w") as f:
        for path in drone_list:
            f.write(path + "\n")


if __name__ == "__main__":
    write_train_test_split_dense_uav()
