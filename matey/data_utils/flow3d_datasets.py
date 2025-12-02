import torch
import torch.nn
import numpy as np
import os
from .blasnet_3Ddatasets import BaseBLASNET3DDataset
import h5py
import json
from operator import mul
from functools import reduce

class Flow3DDataset(BaseBLASNET3DDataset):

    #  cond_field_names = ["cell_types"]
    #  cond_field_names = ["sdf_obstacle"]
    cond_field_names = ["sdf_obstacle", "sdf_channel"]

    @staticmethod
    def _specifics():
        """45 cases of flow around objects:
        Domain    : 0.4m x 0.1m x 0.1m
        Mesh      : 192 x 48 x 48
        Time steps: 5000 [0.1ms each]
        Flow      : speed 20 m/s, Re = 2*10^5
        """
        time_index = 1
        sample_index = 0
        #  field_names = ["Vx", "Vy", "Vw", "Pressure", "k", "nut"]
        field_names = ["Vx", "Vy", "Vw", "Pressure"]
        type = "flow3d"
        cubsizes = [192, 48, 48]
        case_str = "*"
        split_level = "case"
        return time_index, sample_index, field_names, type, cubsizes, case_str, split_level
    field_names = _specifics()[2] #class attributes

    def _get_filesinfo(self, file_paths):
        dictcase = {}
        for datacasedir in file_paths:
            file = os.path.join(datacasedir, "data.h5")
            f = h5py.File(file)
            nsteps = 5000
            features_mapping = {'u': slice(0, 3), 'p': 3}
            features = list(features_mapping.keys())

            assert nsteps == f["data/times"].shape[0]

            # check that the features we map are present in the data
            data_keys = [ key for key in f["data"].keys() if not key == "times" ]
            assert all(key in data_keys for key in features)

            dictcase[datacasedir] = {}
            dictcase[datacasedir]["file"] = file
            dictcase[datacasedir]["sdf"] = os.path.join(datacasedir, "sdf_neg_one.npz")
            dictcase[datacasedir]["ntimes"] = nsteps
            dictcase[datacasedir]["features"] = features
            dictcase[datacasedir]["features_mapping"] = features_mapping

            # Should probably do prior to training. Just for testing right now
            json_path = os.path.join(datacasedir, "stats.json")
            with open(json_path) as fjson:
                try:
                    dictcase[datacasedir]["stats"] = json.load(fjson)
                except:
                    raise RuntimeError("Could not read %s" % json_path)

        return dictcase

    def _reconstruct_sample(self, dictcase, time_idx, leadtime):
        f = h5py.File(dictcase["file"])
        features = dictcase["features"]

        padded_cell_counts = np.array(f["grid/cell_counts"])
        total_padded_cells = reduce(mul, padded_cell_counts)
        inside_idx = np.array(f["grid/cell_idx"])

        def get_data(start, end):
            assert end <= 5000, f'{dictcase["file"]}, end = {end}'
            n_steps = end - start
            n_features = 0
            for ft in features:
                key = f["data"][ft]
                n_features = n_features + (1 if len(key.shape) == 2 else key.shape[2])

            # We don't want to worry about this value affecting normalization
            # later on. If we set it to 1e10, it will screw up with that. So,
            # set it to 0, as is done in the generative-turbulence repository.
            INVALID_VALUE = 0 # FIXME: change to something non-zero
            data = np.full((n_steps, total_padded_cells, n_features), INVALID_VALUE, dtype='f4')

            ft_mapping = dictcase["features_mapping"]
            for ft in features:
                ft_idx = ft_mapping[ft]
                mean = dictcase["stats"][ft]['mean']
                std = dictcase["stats"][ft]['std']
                try:
                    std = [max(x, 1e-4) for x in std]
                except:
                    std = max(std, 1e-4)

                # Set (normalized) inside cell values
                data[:, inside_idx, ft_idx] = (f["data"][ft][start:end] - mean)/std

                # Set (normalized) fixed-value boundary conditions
                bcs = f['boundary-conditions'][ft]
                for name, desc in bcs.items():
                    if desc.attrs['type'] == 'fixed-value':
                        boundary_idx = np.array(f['grid/boundaries'][name])
                        data[:, boundary_idx, ft_idx] = (np.atleast_1d(desc['value']) - mean)/std

            # Set conditional data
            cond_field_names = Flow3DDataset.cond_field_names
            n_cond = len(cond_field_names)

            if cond_field_names[0] == "cell_types":
                assert n_cond == 1
                # Cell types
                CELL_DICT = {'outside': 0, 'inside': 1, 'inlets': -2, 'outlets': -3, 'walls': -4}
                cond_data = np.full((n_steps, total_padded_cells, 1), CELL_DICT['outside'], dtype='f4')
                cond_data[:, inside_idx, :] = CELL_DICT['inside']
                for name in f["grid/boundaries"].keys():
                    cond_data[:, np.array(f["grid/boundaries"][name]), :] = CELL_DICT[name]
            else:
                ## Precomputed signed distance function
                cond_data = np.full((n_steps, total_padded_cells, n_cond), 0, dtype='f4')

                with open(dictcase["sdf"], "rb") as fp:
                    sdf = np.load(fp)
                    for s, sdf_name in enumerate(cond_field_names):
                        sdf_field = sdf[sdf_name][:,0]
                        #  sdf_field = np.where(sdf_field > 0, 1., 0.)
                        for d in range(n_steps):
                            cond_data[d,:,s] = sdf_field

            # Rearrange the data to T,H,W,D,C
            data = data.reshape((n_steps, *padded_cell_counts, n_features))
            cond_data = cond_data.reshape((n_steps, *padded_cell_counts, n_cond))

            # Rearrange the data to T,C,D,H,W
            data = data.transpose((0, -1, 3, 1, 2))
            cond_data = cond_data.transpose((0, -1, 3, 1, 2))

            # We reduce H dimension from 194 x 50 x 50 to 192 x 50 x 50 to
            # allow reasonable patch sizes. Otherwise, as 194 = 2 x 97, it
            # would only allow us patch size of 2 or 97, neither of which are
            # reasonable.

            # Pressure has fixed-value boundary condition on the outflow. If we
            # simply reduce the dimension by eliminating the last two layers,
            # we lose that information. Instead, set the last H layer to be the
            # outflow.
            #  cond_data[:,:,:,-3,:] = cond_data[:,:,:,-1,:]  # only for cell types
            #  data[:,ft_mapping['p'],:,-3,:] = data[:,ft_mapping['p'],:,-1,:]

            return data, cond_data

        comb_x, cond_data = get_data(time_idx, time_idx + self.nsteps_input)
        comb_y, _ = get_data(time_idx + self.nsteps_input + leadtime - 1, time_idx + self.nsteps_input + leadtime)

        comb = np.concatenate((comb_x, comb_y), axis=0)

        return torch.from_numpy(comb), leadtime.to(torch.float32), torch.from_numpy(cond_data)

    def _get_specific_bcs(self):
        # FIXME: not used for now
        return [0, 0, 0]  # Non-periodic
