import numpy as np
from scipy.misc import imread, imsave
import os
from skimage.transform import rescale


def mean_image_shape(target_dir):

    size_sum = np.zeros(2)
    n_images = 0

    path = target_dir
    for img in os.listdir(path):
        f = imread(path+img)
        size_sum += f.shape[:2]
        n_images += 1

    return np.max(np.round(size_sum/n_images))


def extract_plant(plant_img, leaf_segments):

    mask = np.copy(leaf_segments)
    mask[np.sum(mask, 2) != 0] = 1

    plant_img = plant_img*mask

    return plant_img


def crop_void(img):
    mask = np.any(img != (0, 0, 0),2)
    img = img[np.any(mask, 1), :, :]
    mask = mask[np.any(mask, 1), :]
    img = img[:, np.any(mask, 0), :]
    return img


def get_data(img_dir):

    for f in os.listdir(img_dir):
        if f[-7:] == "rgb.png" and f[:-7]+"label.png" in os.listdir(img_dir):
            yield imread(img_dir+f), imread(img_dir+f[:-7]+"label.png")


def extract_plants():
    img_dir = "./Plant_Phenotyping_Datasets/Plant/"
    to_dir = "./gen_dat/"

    for p in os.listdir(img_dir):
        for i, (img, seg) in enumerate(get_data(img_dir + p + '/')):

            plant = extract_plant(img, seg)
            plant = crop_void(plant)

            td = to_dir + p + '/'
            if p not in os.listdir(to_dir):
                os.mkdir(td)

            imsave(td+str(i) + ".png", plant)


def rescale_tobacco():
    subject_path = "./gen_dat/Tobacco/"
    target_path = "./gen_dat/Ara2012/"

    max_mean = mean_image_shape(target_path)

    for im_path in os.listdir(subject_path):
        img = imread(subject_path+im_path)
        img = rescale(img, max_mean/np.max(img.shape[:2]), mode="reflect")
        imsave(subject_path+im_path, img)


if __name__ == "__main__":

    extract_plants()
    rescale_tobacco()
