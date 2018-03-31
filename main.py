import numpy as np
from scipy.misc import imread, imsave
import os
from matplotlib import pyplot as plt
import noise

def extract_plant(plant_img, leaf_segments):

    mask = np.copy(leaf_segments)
    #print(np.sum(mask, 1)
    mask[np.sum(mask, 2) != 0] = 1

    no_background = plant_img*mask

    leaves = []

    matcher = leaf_segments + np.array([0, 256, 256*2])
    matcher = np.sum(matcher, 2)
    unique_seg = np.unique(matcher)
    unique_seg = unique_seg[unique_seg != (0 or 256*3)]

    for i, u in enumerate(unique_seg):

        tmp = np.copy(no_background)
        tmp[matcher != u] = 0
        leaves.append(tmp)

    return leaves


def get_data(img_dir):

    imgs = {}
    seg = {}

    for f in os.listdir(img_dir):
        if f[-7:] == "rgb.png":
            imgs[f[:-8]] = imread(img_dir+f)

        elif f[-9:] == "label.png":
            seg[f[:-10]] = imread(img_dir+f)

    return imgs, seg


def extract_plants():
    img_dir = "./Plant_Phenotyping_Datasets/Plant/Ara2012/"
    to_dir = "./gen_dat/ara/"

    images, segment = get_data(img_dir)
    pi = {}

    for im in images.keys():
        if im in images and im in segment:
            pi[im] = extract_plant(images[im], segment[im])


    for i, im in enumerate(images.keys()):
        if im in images and im in segment:
            if str(i) in os.listdir(to_dir):
                if len(os.listdir(to_dir+str(i))) != 0:
                    [os.remove(to_dir+str(i)+"/"+x) for x in os.listdir(to_dir+str(i))]
                os.rmdir(to_dir+str(i))
            os.mkdir(to_dir+str(i))
            for j, leaf in enumerate(pi[im]):
                imsave(to_dir+str(i)+"/"+str(j)+".png", leaf)


def gen_noise(shape):
    shape = shape
    scale = 40
    octaves = 5
    persistence = 0.6
    lacunarity = 3.0

    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(i / scale,
                                        j / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=1024,
                                        repeaty=1024,
                                        base=0)

    world += np.min(world)*-1
    world /= np.max(world)
    ret = np.zeros((shape[0], shape[1], 3))
    ret += np.array([250, 175, 100])/265#[204, 138, 46])/256
    print(np.max(world), np.min(world))
    ret *= world.reshape(shape[0], shape[1], 1)

    return ret


def place_plants_structured(shape, col_spacing=None, row_spacing=None):

    tobacco_prob = 0.00

    col_spacing = 20 if col_spacing is None else col_spacing
    row_spacing = 20 if row_spacing is None else row_spacing

    n_ara = len(os.listdir("./gen_dat/ara/"))
    n_tobacco = len(os.listdir("./gen_dat/tobacco/"))

    ret = np.zeros(shape)
    for y in range(row_spacing, shape[0], row_spacing):
        for x in range(col_spacing, shape[1], col_spacing):
            if np.random.rand() > tobacco_prob:
                ret[y, x] = 1
            else:
                ret[y, x] = 2

    return ret


def draw_on_plant(img, coords, plant_path):

    for i in os.listdir(plant_path):
        p_img = imread(plant_path+i)
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




def draw_on_plants(plant_map, ara_path, tobacco_path):

    ret = np.zeros(plant_map.shape+(3,))

    aras = os.listdir(ara_path)
    tobaccos = os.listdir(tobacco_path)

    for i in np.argwhere(plant_map != 0):
        if plant_map[i[0], i[1]] == 1:
            draw_on_plant(ret, i, ara_path+aras[np.random.randint(0, len(aras))]+'/')
        elif plant_map[i[0], i[1]] == 2:
            draw_on_plant(ret, i, tobacco_path+tobaccos[np.random.randint(0, len(tobaccos))]+'/')

    return ret

ara_path = "./gen_dat/ara/"
tobacco_path = "./gen_dat/tobacco/"
dirt = gen_noise((1000, 1000)) * 265

img = place_plants_structured((1000, 1000), 200, 200)
img = draw_on_plants(img, ara_path, tobacco_path)

img /= 256
img[(img[:, :, 0] != 0) & (img[:, :, 1] != 0) & (img[:, :, 2] != 0)] = 1

imsave("./test.png", img)
plt.imshow(img)
plt.show()
