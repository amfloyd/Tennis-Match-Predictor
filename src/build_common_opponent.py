import pandas as pd
import numpy as np
from feature_extractor import common_opponent_feature_extractor
# import feature_extractor


if __name__ == "__main__":

    '''
        This is not a main script for the project. It is only meant to help generate the common opponent features on
        a remote machine. This way our laptops don't get overworked
    '''

    player_specific_features = [
        "ace",
        "df",
        "svpt",
        "1stIn",
        "1stWon",
        "2ndWon",
        "SvGms",
        "bpFaced",
        "bpSaved"
    ]

    data_types = ["train", "valid", "test"]
    cof = {
        "train": {f: None for f in player_specific_features},
        "valid": {f: None for f in player_specific_features},
        "test": {f: None for f in player_specific_features}
    }

    for data_type in data_types:
        # Load data
        data_file = "../data/splits/{0}.csv".format(data_type)
        data = pd.read_csv(data_file, encoding="utf8")

        for f in player_specific_features:
            cof[data_type][f] = common_opponent_feature_extractor(data, f, data_type)
            r = open("../results/{0}_{1}.txt".format(data_type, f),'w')
            r.write("Done")
            r.close()

    print("stop")

