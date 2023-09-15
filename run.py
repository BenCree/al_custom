# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point for running a single cycle of active learning."""
import time
import re
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

import sars_test.utils
from al_for_fep.configs.simple_greedy_gaussian_process import get_config as get_gaussian_process_config
import ncl_cycle
from ncl_cycle import ALCycler

from concurrent.futures import ProcessPoolExecutor

from pathlib import Path
import time
import re

oracle = pd.read_csv("gen_metrics.csv")
oracle.sort_values(by='cnnaff', ascending=False, inplace=True)
# find 5% best values cutoff
cutoff = oracle[:int(0.05*len(oracle))].cnnaff.values[0]

import pandas as pd


def ask_oracle(chosen_ones, virtual_library, col):
    # check and return all the values for the smiles
    # look up and overwrite the values in place

    # look the up by smiles
    oracle_has_spoken = chosen_ones.merge(oracle, on=['Smiles'])
    # get the correct affinities
    print(oracle_has_spoken)
    chosen_ones.cnnaff = -oracle_has_spoken[f'{col}'].values
    assert np.all(chosen_ones.Smiles.values == oracle_has_spoken.Smiles.values)
    # update the main dataframe
    virtual_library.update(chosen_ones)

def report(virtual_library, start_time, col, dat, cycle_id):
    # select only the ones that have been chosen before
    best_finds = virtual_library[virtual_library[col] < -5]  #-6 is about 5% of the best cases
    print(f"IT: {cycle_id},Lib size: {len(virtual_library)}, "
          f"training size: {len(virtual_library[virtual_library.Training])}, "
          #f"cnnaff 0: {len(virtual_library[virtual_library.cnnaff == 0])}, "
          #f"<-6 cnnaff_per_mw: {len(best_finds)}, "
          f"time: {time.time() - start_time}, "
          f"average {col}: {chosen_ones[col].mean()}")



    # Calculate statistics
    lib_size = len(virtual_library)
    training_size = len(virtual_library[virtual_library[
        'Training']])  # Replace 'Training' with your actual column name for training data if different
    num_best_finds = len(best_finds)
    elapsed_time = time.time() - start_time
    avg_cnn_per_mw = chosen_ones[col].mean()  # Assuming best_finds DataFrame is not empty. Handle empty case appropriately.

    # Add new metrics as columns
    dat[f'Lib_size'] = [lib_size]
    dat[f'Training_size'] = [training_size]
    dat[f'<-6_cnnaff_per_mw'] = [num_best_finds]
    dat[f'Time'] = [elapsed_time]
    dat[f'Avg_cnnaff_per_mw'] = [avg_cnn_per_mw]
    dat[f'Cycle'] = [cycle_id]
    #dat[f'Target_cnn_to_beat'] = 0.015625
    dat.to_csv(f'{col}_dat.csv', )

    #dat.set_index(cycle_id, inplace=True)

    # Concatenate along rows


    return dat


if __name__ == '__main__':
    models = ['rf', 'mlp', 'gp', 'linear', 'elasticnet', 'gbm']
    bits = [1024, 2048]
    feats = ['fingerprint', 'fingerprints_and_descriptors']
    init_ns = [25, 50, 100, 200]
    init_selection_methods = ['none', 'maxmin']
    sfs = ['logp_cnn_per_mw', 'qed_cnn_per_mw', 'sa_cnn_per_mw', 'cnn_per_mw']
    stime = time.time()
    from concurrent.futures import ThreadPoolExecutor
    import pandas as pd
    from pathlib import Path
    import time
    import re

    parameters = []
    for sf in sfs:
        for feat in feats:
            for bit in bits:
                for model in models:
                    for init_n in init_ns:
                        parameters.append((sf, feat, bit, model, init_n))


    def perform_cycle(params):
        sf, feat, bit, model, init_n = params

        dat = pd.DataFrame()
        output = Path(f'{sf}_{feat}_{model}_{init_n}_cycle')
        col = sf
        previous_trainings = list(map(str, output.glob('cycle_*/selection.csv')))

        dat = pd.DataFrame()
        output = Path(f'{sf}_{feat}_{model}_{init_n}_cycle')
        col = sf
        previous_trainings = list(map(str, output.glob('cycle_*/selection.csv')))
        print('Loading previous cycles:', previous_trainings)

        config = get_gaussian_process_config()
        config.model_config.model_type = model
        config.model_config.features.feature_type = feat
        config.model_config.features.params.fingerprint_size = bit
        config.training_pool = ','.join([f"initial_{sf}.csv"] + previous_trainings)
        config.virtual_library = f"gen_metrics_{sf}.csv"
        config.selection_config.num_elements = init_n    # how many new to select
        config.selection_config.selection_columns = ["Smiles", "id", "name", "cnnaff_pos", "cnnaff", "MW", "HBA", \
                                                     "HBD", "LogP", "Pass_Ro5", "has_pains", "has_unwanted_subs", \
                                                     "has_prob_fgs", "synthetic_accessibility", \
                                                     "TPSA", "QED", "N_RotBonds"] + [sf]

        config.model_config.targets.params.feature_column = col

        AL = ALCycler(config)
        virtual_library = AL.get_virtual_library()

        cycle_start = 0
        if previous_trainings:
            cycle_start = max(int(re.findall("[\d]+", cycle)[0]) for cycle in previous_trainings)
        cycle_dat = pd.DataFrame()
        for cycle_id in range(cycle_start, 2):
            start_time = time.time()
            chosen_ones, virtual_library_regression = AL.run_cycle(virtual_library)

            # the new selections are now also part of the training set
            virtual_library_regression.loc[chosen_ones.index, ncl_cycle.TRAINING_KEY] = True
            ask_oracle(chosen_ones, virtual_library_regression, col)  # TODO no conformers? penalise
            virtual_library = virtual_library_regression

            # expand the virtual library
            # if len(virtual_library[virtual_library.Smiles == "CN(C(=O)c1cn(C)nc1-c1ccc(F)cc1F)c1nc2ccccc2n1C"]) == 0:
            #     new_record = pd.DataFrame([{'Smiles': "CN(C(=O)c1cn(C)nc1-c1ccc(F)cc1F)c1nc2ccccc2n1C", ncl_cycle.TRAINING_KEY: False}])
            #     expanded_library = pd.concat([virtual_library_regression, new_record], ignore_index=True)
            #     virtual_library = expanded_library

            cycle_dir = Path(f"{output}/cycle_{cycle_id:04d}")
            cycle_dir.mkdir(exist_ok=True, parents=True)
            virtual_library.to_csv(cycle_dir / 'virtual_library_with_predictions.csv', index=False)
            chosen_ones.to_csv(cycle_dir / "selection.csv", columns=config.selection_config.selection_columns, index=False)

            dat = report(virtual_library, start_time, col, dat, cycle_id)
            cycle_dat = pd.concat([cycle_dat, dat],)

            cfg_json = config.to_json()
            with open(f'{cycle_dir}/config.json', 'w') as fout:
                fout.write(cfg_json)

        cycle_dat.to_csv(f'{output}/{cycle_id}_dat.csv')


    with ProcessPoolExecutor(max_workers=15) as executor:
        executor.map(perform_cycle, parameters)

ftime = time.now()

print(f'Hyperparam run took: {ftime - stime} s')
