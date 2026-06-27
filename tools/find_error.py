import os
import filecmp
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageChops
from tqdm import tqdm


def _images_are_identical(image_a: Path, image_b: Path) -> bool:
	"""Return True when two image files have identical RGB pixel content."""
	if _files_are_byte_identical(image_a, image_b):
		return True

	with Image.open(image_a) as img_a, Image.open(image_b) as img_b:
		if img_a.size != img_b.size:
			return False

		rgb_a = img_a.convert("RGB")
		rgb_b = img_b.convert("RGB")

		diff = ImageChops.difference(rgb_a, rgb_b)
		return diff.getbbox() is None


def _files_are_byte_identical(file_a: Path, file_b: Path) -> bool:
	"""Fast exact-file check used as a shortcut before decoding images."""
	try:
		if file_a.stat().st_size != file_b.stat().st_size:
			return False
	except OSError:
		return False

	return filecmp.cmp(file_a, file_b, shallow=False)


def _image_pixel_signature(image_path: Path) -> Optional[Tuple[Tuple[int, int], str]]:
	"""Build a content signature from decoded RGB pixels."""
	try:
		with Image.open(image_path) as img:
			rgb = img.convert("RGB")
			raw = rgb.tobytes()
			digest = hashlib.sha1(raw).hexdigest()
			return rgb.size, digest
	except OSError:
		return None


def find_duplicate_images_in_current_root(
	current_root: str,
	image_suffixes: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
) -> List[List[str]]:
	"""Find duplicate images inside one folder and return duplicate filename groups."""
	folder = Path(current_root)
	if not folder.exists() or not folder.is_dir():
		raise FileNotFoundError(f"Folder not found: {current_root}")

	signature_to_files: Dict[Tuple[Tuple[int, int], str], List[str]] = {}
	for entry in folder.iterdir():
		if not entry.is_file():
			continue
		if entry.suffix.lower() not in image_suffixes:
			continue

		sig = _image_pixel_signature(entry)
		if sig is None:
			continue

		signature_to_files.setdefault(sig, []).append(entry.name)

	duplicate_groups: List[List[str]] = []
	for files in signature_to_files.values():
		if len(files) > 1:
			duplicate_groups.append(sorted(files))

	duplicate_groups.sort(key=lambda g: (g[0], len(g)))
	return duplicate_groups


def list_folders_with_internal_duplicate_images(
	root_dir: str = "/media/data1/feihong/remote/data/drone_img",
	image_suffixes: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
) -> Dict[str, List[List[str]]]:
	"""Scan all subfolders and return those that contain duplicate images."""
	root_path = Path(root_dir)
	if not root_path.exists():
		raise FileNotFoundError(f"Root directory not found: {root_dir}")

	folder_list: List[Path] = []
	for current_root, _, _ in os.walk(root_path):
		folder_list.append(Path(current_root))

	result: Dict[str, List[List[str]]] = {}
	for folder in tqdm(folder_list, desc="Checking internal duplicates", unit="folder"):
		groups = find_duplicate_images_in_current_root(
			current_root=str(folder),
			image_suffixes=image_suffixes,
		)
		if groups:
			result[str(folder)] = groups

	return result


def list_folders_with_same_images(
	root_dir: str = "/media/data1/feihong/remote/data/drone_img",
	image_name_a: str = "300_135.png",
	image_name_b: str = "300_225.png",
) -> List[str]:
	"""List folders where image_name_a and image_name_b both exist and are identical."""
	root_path = Path(root_dir)
	if not root_path.exists():
		raise FileNotFoundError(f"Root directory not found: {root_dir}")

	matched_folders: List[str] = []
	candidate_folders: List[Path] = []

	for current_root, _, files in os.walk(root_path):
		file_set = set(files)
		if image_name_a not in file_set or image_name_b not in file_set:
			continue
		candidate_folders.append(Path(current_root))

	for folder_path in tqdm(candidate_folders, desc="Comparing image pairs", unit="folder"):
		image_a = folder_path / image_name_a
		image_b = folder_path / image_name_b

		try:
			if _images_are_identical(image_a, image_b):
				matched_folders.append(str(folder_path))
		except OSError:
			# Skip unreadable/corrupted images and continue scanning.
			continue

	matched_folders.sort()
	return matched_folders


def main() -> None:
	# folders = list_folders_with_same_images()
	duplicate_groups = list_folders_with_internal_duplicate_images()
	for folder in duplicate_groups:
		print(folder)
	# for folder, groups in duplicate_groups.items():
	# 	print(folder)
	# 	for group in groups:
	# 		print(f"  {group}")
	# print(f"Total matched folders: {len(folders)}")


if __name__ == "__main__":
	main()
