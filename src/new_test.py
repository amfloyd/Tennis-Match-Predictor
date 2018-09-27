import pandas as pd 
import numpy as np 
import math 
import os

from baselines import extract_win_cnts, matches_played
from feature_extractor import generate_win_ratio_feature 

def gen_test_frame(train_frame, test_data_type):
    # train_file = '../data/splits/train.csv'
    # valid_file = '../data/splits/valid.csv'
    # test_file = '../data/splits/test.csv'

    load_file = "mod_{0}.csv".format(test_data_type)
    if os.path.exists(load_file):
        return pd.read_csv(load_file, encoding="utf8")
    else:

        test_file = '../data/splits/{0}.csv'.format(test_data_type)


        # raw_train = pd.read_csv(train_file, encoding='utf8')
        # raw_val = pd.read_csv(valid_file,encoding='utf8')
        raw_test = pd.read_csv(test_file, encoding='utf8')

        # raw_data = raw_train
        # raw_data = pd.concat([raw_train, raw_val])

        player_names = pd.concat([train_frame["player_1_name"], train_frame["player_2_name"]]).unique()

        raw_features = [
            "ace",
            "df",
            "svpt",
            "1stIn",
            "1stWon",
            "2ndWon",
            "SvGms",
            "bpFaced",
            "bpSaved",
            "seed"
        ]
        extra_features = [
            "h2h",
            "rank",
            "win_ratio",
            "completeness"
        ]

        win_hist = extract_win_cnts(train_frame)
        match_hist = matches_played(train_frame)


        p_info = {}
        p1 = "player_1_"
        p2 = "player_2_"
        for pname in player_names:
            p1_stats = train_frame[(train_frame["player_1_name"]==pname)]
            p2_stats = train_frame[(train_frame["player_2_name"]==pname)]
            stats = pd.concat([p1_stats, p2_stats])
            test_feats = {}
            p_info[pname] = test_feats
            for f in raw_features:
                p1_feat = np.reshape(p1_stats[p1+f].values, [len(p1_stats[p1+f].values),1])
                p2_feat = np.reshape(p2_stats[p2+f].values, [len(p2_stats[p2+f].values),1])
                avg_feat = np.mean(np.nan_to_num(np.concatenate((p1_feat, p2_feat), axis=0)))
                test_feats[f] = avg_feat
            for f in extra_features:
                if f == "h2h":
                    test_feats[f] = win_hist[pname]
                elif f=="rank":
                    p1_dates = np.reshape(p1_stats["tourney_date"].values, [len(p1_stats[p1+f].values),1])
                    p2_dates = np.reshape(p2_stats["tourney_date"].values, [
                                            len(p2_stats[p2+f].values), 1])
                    all_dates = np.nan_to_num(
                        np.concatenate((p1_dates, p2_dates), axis=0))
                    latest = np.max(all_dates.astype(int))
                    match = stats[(stats["tourney_date"] == latest)]
                    if match["player_1_name"].values[0] == pname:
                        rank = match["player_1_rank"]
                    else:
                        rank = match["player_2_rank"]
                    test_feats[f] = rank.values[0]
                else:
                    test_feats[f] = win_hist[pname]/match_hist[pname]

            test_feats["completeness"] = test_feats["1stWon"] * test_feats["2ndWon"] * (test_feats["bpSaved"]/ test_feats["bpFaced"])
        # write to file  p_info

        features = raw_features
        features.extend(extra_features)

        # Iterate through the test frame to generate the features
        new_feats = pd.DataFrame([[0 for f in features]], columns=features)
        for row in raw_test.itertuples():
            temp_df = pd.DataFrame([[0 for f in features]], columns=features)
            for f in features:
                p1_feature, p2_feature = "player_1_{0}".format(f), "player_2_{0}".format(f)
                p1_name, p2_name = row.player_1_name, row.player_2_name

                if p1_name not in p_info and p2_name in p_info:
                    temp_df[f] = 0 - p_info[p2_name][f]
                elif p1_name in p_info and p2_name not in p_info:
                    temp_df[f] = p_info[p1_name][f] - 0
                elif p1_name not in p_info and p2_name not in p_info:
                    temp_df[f] = 0
                else:
                    temp_df[f] = p_info[p1_name][f] - p_info[p2_name][f]



                # temp_df[f] = raw_test[p1_feature] - raw_test[p2_feature]

            new_feats = new_feats.append(temp_df, ignore_index=True)

        new_feats.drop(new_feats.index[0])

        print("stop")
        new_feats.to_csv("mod_{0}.csv".format(test_data_type), encoding="utf8")
        return new_feats


if __name__ == "__main__":

    raw_train = pd.read_csv('../data/splits/train.csv', encoding='utf8')
    a = gen_test_frame(raw_train, "test")
