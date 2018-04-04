import numpy as np
from scipy.misc import imread, imsave
import os
import noise
import sys
from skimage.transform import rescale
from multiprocessing import Process, Value
from ctypes import c_int32
from time import sleep


def gen_noise(shape):
    shape = shape
    scale = 40
    octaves = 5
    persistence = 0.6
    lacunarity = 3.0

    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                world[i][j][k] = noise.pnoise3(i / scale,
                                            j / scale,
                                            k / scale,
                                            octaves=octaves,
                                            persistence=persistence,
                                            lacunarity=lacunarity,
                                            repeatx=1024,
                                            repeaty=1024,
                                            base=0)

    world += np.min(world)*-1
    world /= np.max(world)
    ret = np.zeros((shape[0], shape[1], 3))
    ret += (np.array([250, 175, 100])/256)*.7

    ret *= world

    return ret


def place_plants_structured(shape, plant_distance, plant_dispersion, tobacco_prob=.05):

    col_spacing = plant_distance[1]
    row_spacing = plant_distance[0]

    ret = np.zeros(shape)
    for y in range(row_spacing//2, shape[0], row_spacing):
        for x in range(col_spacing//2, shape[1], col_spacing):
            pos = plant_dispersion(np.array((y, x)))
            if np.random.rand() > tobacco_prob:
                ret[pos[0], pos[1]] = 1
            else:
                ret[pos[0], pos[1]] = 2

    return ret


def draw_on_plant(img, coords, plant_path, scale=1):

    p_img = imread(plant_path)
    if scale != 1:
        p_img = rescale(p_img, scale, mode='reflect')
    idx_from = [int(coords[j]-p_img.shape[j]//2) for j in range(2)]
    from_cut_off = [0 if idx_from[j] > 0 else np.abs(idx_from[j]) for j in range(2)]
    idx_to = [int(coords[j]+p_img.shape[j]//2)+(p_img.shape[j]%2) for j in range(2)]
    to_cut_off = [0 if idx_to[j] < img.shape[j] else idx_to[j]-img.shape[j] for j in range(2)]

    yf = idx_from[0]+from_cut_off[0]
    yt = idx_to[0] - to_cut_off[0]
    xf = idx_from[1]+from_cut_off[1]
    xt = idx_to[1] - to_cut_off[1]

    p_paste = p_img[from_cut_off[0]:-to_cut_off[0] if to_cut_off[0] != 0 else None,
                        from_cut_off[1]:-to_cut_off[1] if to_cut_off[1] != 0 else None]

    img[yf:yt, xf:xt][p_paste != 0] = p_paste[p_paste != 0]


def draw_on_feature(f_map, coords, plant_path, feature, scale=1):

    p_img = imread(plant_path)
    if scale != 1:
        p_img = rescale(p_img, scale, mode='reflect')

    idx_from = [int(coords[j]-p_img.shape[j]//2) for j in range(2)]
    from_cut_off = [0 if idx_from[j] > 0 else np.abs(idx_from[j]) for j in range(2)]
    idx_to = [int(coords[j]+p_img.shape[j]//2)+(p_img.shape[j]%2) for j in range(2)]
    to_cut_off = [0 if idx_to[j] < f_map.shape[j] else idx_to[j]-f_map.shape[j] for j in range(2)]

    yf = idx_from[0]+from_cut_off[0]
    yt = idx_to[0] - to_cut_off[0]
    xf = idx_from[1]+from_cut_off[1]
    xt = idx_to[1] - to_cut_off[1]

    p_paste = p_img[from_cut_off[0]:-to_cut_off[0] if to_cut_off[0] != 0 else None,
                    from_cut_off[1]:-to_cut_off[1] if to_cut_off[1] != 0 else None]

    f_map[yf:yt, xf:xt][np.any(p_paste != 0, 2)] = feature


def draw_on_plants(plant_map, ara_path, tobacco_path, scale_fn):

    ret_img = np.zeros(plant_map.shape+(3,))
    ret_feature = np.zeros(plant_map.shape)

    aras = os.listdir(ara_path)
    tobaccos = os.listdir(tobacco_path)

    plant_coords = np.argwhere(plant_map != 0)
    z_index = np.arange(plant_coords.shape[0])
    np.random.shuffle(z_index)

    for i in z_index:
        coords = plant_coords[i]
        if plant_map[coords[0], coords[1]] == 1:
            plant = ara_path+aras[np.random.randint(0, len(aras))]
            scale = scale_fn()
            draw_on_plant(ret_img, coords, plant, scale)
            draw_on_feature(ret_feature, coords, plant, 0, scale=scale)
        elif plant_map[coords[0], coords[1]] == 2:
            plant = tobacco_path+tobaccos[np.random.randint(0, len(tobaccos))]
            scale = scale_fn()
            draw_on_plant(ret_img, coords, plant, scale)
            draw_on_feature(ret_feature, coords, plant, 1, scale=scale)

    return ret_img, ret_feature


def gen_n_images(id_range, generator_fn, counter):

    for i in range(id_range[0], id_range[1]):
        img, feature = generator_fn()
        imsave(output_dir + "data/" + str(i) + ".png", img)
        imsave(output_dir + "feature/" + str(i) + ".png", feature)
        with counter.get_lock():
            counter.value += 1


if __name__ == '__main__':

    # hyper params
    n_threads = 10
    n_images = 1000

    # params
    distance = 10
    weed_probability = .02
    img_size = (360, 640)
    plant_distance = (np.array((100, 150)) * (1/distance)).astype(np.int64)
    plant_distance_dispersion = .1*plant_distance
    plant_scale = .5 * (1/distance)
    plant_scale_dispersion = .1*plant_scale

    ara_path = "./gen_dat/Ara2012/"
    tobacco_path = "./gen_dat/Tobacco/"
    output_dir = "./data_set/{}/".format(distance)

    # make output dir if !exists
    if str(distance) not in os.listdir("./data_set/"):
        os.mkdir(output_dir)
        os.mkdir(output_dir+"data/")
        os.mkdir(output_dir+"feature/")

    index = len(os.listdir(output_dir+"data/"))

    def plant_pos_dispersion_fn(pos):
        ret = (plant_distance_dispersion * np.random.randn(2) + pos).astype(np.int64)
        ret[ret < 0] = 0
        ret[ret >= img_size] = (np.array(img_size)-1)[ret >= img_size]
        return ret

    def plant_scale_fn():
        ret = plant_scale_dispersion*np.random.randn()+plant_scale
        return ret if ret > 0 else 0

    def parametrized_image_gen():
        dirt = gen_noise(img_size + (3,))
        plant_placements = place_plants_structured(img_size, plant_distance, plant_pos_dispersion_fn,
                                                   tobacco_prob=weed_probability)
        img, feature = draw_on_plants(plant_placements, ara_path, tobacco_path, plant_scale_fn)
        mask = np.all(np.equal(img[:, :], (0, 0, 0)), 2)
        img[mask] = dirt[mask]
        return img, feature


    n_images_per_thread = n_images // n_threads
    threads = []
    img_counter = Value(c_int32)
    for i in range(n_threads):
        im_range = (n_images_per_thread*i+index, n_images_per_thread*(i+1) +
                    (n_images % n_threads if i == (n_threads - 1) else 0) + index)
        threads.append(Process(target=gen_n_images, args=(im_range, parametrized_image_gen, img_counter)))

        threads[-1].start()

    while any([t.is_alive() for t in threads]):
        print("\r{0} of {1} images generated".format(img_counter.value, n_images), end='')
        sleep(1)




