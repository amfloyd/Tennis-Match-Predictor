import pandas as pd 
import numpy as np 
import math 

from baselines import extract_win_cnts

train_file = '../data/splits/train.csv'
valid_file = '../data/splits/valid.csv'
test_file = '../data/splits/test.csv'


raw_train = pd.read_csv(train_file, encoding='utf8')
raw_val = pd.read_csv(valid_file,encoding='utf8')


raw_data = raw_train
# raw_data = pd.concat([raw_train, raw_val])

player_names = pd.concat([raw_data["player_1_name"], raw_data["player_2_name"]]).unique()

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
    "win_ratio"
]

win_hist = extract_win_cnts(raw_data)

p_info = {}
for pname in player_names:
    p1_stats = raw_data[(raw_data["player_1_name"]==pname)]
    p2_stats = raw_data[(raw_data["player_2_name"]==pname)]
    stats = pd.concat([p1_stats, p2_stats])
    p1 = "player_1_"
    p2 = "player_2_"
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
            test_feats[f] = rank

    # else: 
