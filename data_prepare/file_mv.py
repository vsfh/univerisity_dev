from glob import glob
import shutil


def list_files():
    # for img_path in glob('D:\\intern\\University-Release\\test\\gallery_drone\\*\\image-01.jpeg'):
    for img_path in glob('D:\\intern\\University-Release\\train\\drone\\*\\image-01.jpeg'):
        img_name = img_path.split('\\')[-2]
        output_path = fr'first_site/{img_name}.jpeg'
        shutil.copy(img_path, output_path)

list_files()