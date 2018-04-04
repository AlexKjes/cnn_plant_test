import numpy as np
import os
from scipy.ndimage import imread


class DataStore:

    def __init__(self, buffer_size, n_iterations_before_reload_data=1):

        self.data_path = "./data_set/"
        self.max_data_itrs = n_iterations_before_reload_data

        self.data_indices = self._map_data()
        self.buffer_size = buffer_size
        self._get_data()

        self.data_index = 0
        self.iteration_index = 0

    def _get_data(self):

        self.data = []
        self.features = []

        keys = list(self.data_indices.keys())
        for i in range(self.buffer_size):
            k = keys[np.random.randint(0, len(keys))]
            i_rand = str(np.random.randint(0, self.data_indices[k]))
            self.data.append(imread(self.data_path+k+'/data/'+i_rand+'.png')/256)
            self.features.append(imread(self.data_path + k + '/feature/' + i_rand+'.png')/256)

        self.data_index = 0
        self.iteration_index = 0


    def _map_data(self):

        ret = {}
        for i in os.listdir(self.data_path):
            ret[i] = len(os.listdir(self.data_path+i+'/data/'))

        return ret

    def get_batch(self, size):

        ret_d = []
        ret_f = []

        for i in range(size):
            ret_d.append(self.data[self.data_index])
            ret_f.append(self.features[self.data_index])

            self.data_index += 1
            if self.data_index == self.buffer_size:
                self.data_index = 0
                self.iteration_index += 1
                if self.max_data_itrs == self.iteration_index:
                    self._get_data()

        return np.array(ret_d), np.array(ret_f)


if __name__ == '__main__':

    data = DataStore(50)
    while True:
        data.get_batch(2)
