import pandas as pd
import numpy as np
import math
import os
import pickle

players_rank_history = pickle.load( open("rank_history.pkl", "rb" ) )
#print (players_rank_history)
players_match_history= pickle.load( open("total_match.pkl", "rb" ) )
#print(players_match_history)
train_player_hist=pickle.load( open("win_history.pkl", "rb" ) )

def generate_head2head(matchups, data_type):

    '''
        The head2head feature should return a ratio of player_1's wins versus player_2's wins
    '''
    train_file = '../data/feature_files/h2h/h2h_train.csv'
    valid_file = '../data/feature_files/h2h/h2h_valid.csv'
    test_file = '../data/feature_files/h2h/h2h_test.csv'

    if data_type == "train" and os.path.exists(train_file):
        df = pd.read_csv(train_file, encoding="utf8")
        return df['h2h']
    elif data_type == "valid" and os.path.exists(valid_file):
        df = pd.read_csv(valid_file, encoding="utf8")
        return df['h2h']
    elif data_type == "test" and os.path.exists(test_file):
        df = pd.read_csv(test_file, encoding="utf8")
        return df['h2h']
    else:
        p1 = matchups["player_1_name"]
        p2 = matchups["player_2_name"]

        p = p1 if p1.nunique() > p2.nunique() else p2

        h2h = []
        # for i in range(len(matchups)):
        for row in matchups.itertuples():

            p1_name = row.player_1_name
            p2_name = row.player_2_name

            p1_wins = 0
            p2_wins = 0

            m = matchups[((matchups["player_1_name"] == p1_name) & (matchups["player_2_name"] == p2_name)) | ((matchups["player_1_name"] == p2_name) & (matchups["player_2_name"] == p1_name)) ]
            # Get the straight matchups
            straight = m[(m["player_1_name"] == p1_name) & (m["player_2_name"] == p2_name)]
            p1_wins += len(straight[straight["player_1_win"] == 1])
            p2_wins += len(straight[straight["player_1_win"] == 0])

            # Get the reverse matchups
            reverse = m[(m["player_1_name"] == p2_name) & (m["player_2_name"] == p1_name)]
            p1_wins += len(reverse[reverse["player_1_win"] == 1])
            p2_wins += len(reverse[reverse["player_1_win"] == 0])

            if p1_wins == 1 and p2_wins == 0:
                # Player 1 will always win
                percentage = 1
            elif p1_wins == 0 and p2_wins == 1:
                # Player 1 has no chance of winning
                percentage = 0
            elif p1_wins == 0 and p2_wins == 0:
                # Player 1 has 50/50 chance of winning
                percentage = .5
            else:
                percentage = p1_wins / (p1_wins + p2_wins)
            h2h.append(percentage)

        print('h2h done')

        h2h_frame = pd.DataFrame(h2h, columns=["h2h"])
        if data_type == "train" and not os.path.exists(train_file):
            h2h_frame.to_csv(train_file, encoding='utf8')
        elif data_type == "valid" and not os.path.exists(valid_file):
            h2h_frame.to_csv(valid_file, encoding='utf8')
        elif data_type == "test" and not os.path.exists(test_file):
            h2h_frame.to_csv(test_file, encoding='utf8')

        return h2h_frame

def generate_set_score_feature(set_scores):
    '''

    :param set_scores: DataFrame column
    :return: DataFrame with one-hot encoding


    Classes:
        - Straight sets (best of 3)
        - Full match (best of 3)
        - Straight sets (best of 5)
        - Four sets (best of 5)
        - Full match (best of 5)
        - Retirement (best of 3 or 5)

        1) Differentiate between full_bo3 and straight_bo5
        2) Handle retirements

        Consider nan as walkover?


    '''

    columns = [
        "straight_bo3",
        "full_bo3",
        "straight_bo5",
        "four_bo5",
        "full_bo5",
        "retirement",
        "walkover"
    ]
    encoding = []

    for score in set_scores:
        if score == "W/O":
            encoding.append([0, 0, 0, 0, 0, 0, 1])
        elif type(score) == float and math.isnan(score):
            encoding.append([0, 0, 0, 0, 0, 0, 1])

        else:
            sets = score.lstrip().rstrip().split(' ')

            if "RET" in sets:
                # Retired
                encoding.append([0, 0, 0, 0, 0, 1, 0])
            elif len(sets) == 5:
                # Full match, best of 5
                encoding.append([0, 0, 0, 0, 1, 0, 0])
            elif len(sets) == 4:
                # Four sets, best of 5
                encoding.append([0, 0, 0, 1, 0, 0, 0])
            elif len(sets) == 3:
                first, second, third = sets[0], sets[1], sets[2]

                first_condition = first[0] > first[2]
                second_condition = second[0] > second[2]
                third_condition = third[0] > third[2]

                if first_condition and second_condition and third_condition:
                    # Straight sets, best of 5
                    encoding.append([0, 0, 1, 0, 0, 0, 0])
                else:
                    # Full match, best of 3
                    encoding.append([0, 1, 0, 0, 0, 0, 0])

            elif len(sets) == 2:
                # Straight sets, best of 3
                encoding.append([1, 0, 0, 0, 0, 0, 0])

    return pd.DataFrame(encoding, columns=columns)

def generate_surface_feature(surface_frame):

    encoding = []
    columns = ["clay", "hard", "grass", "carpet"]

    for s in surface_frame:
        if s == "Clay":
            encoding.append([1,0,0,0])
        elif s == "Hard":
            encoding.append([0,1,0,0])
        elif s == "Grass":
            encoding.append([0,0,1,0])
        else:
            # Carpet
            encoding.append([0,0,0,1])

    return pd.DataFrame(encoding, columns=columns)


def common_opponent_feature_extractor(raw_data, feature_name, data_type):
    data_file = '../data/feature_files/{0}/{0}_{1}.csv'.format(feature_name, data_type)

    if data_type == "train" and os.path.exists(data_file):
        df = pd.read_csv(data_file, encoding="utf8")
        return df[feature_name]
    else:
        print("{0} does not exist".format(data_file))

        p1_feature = "player_1_{0}".format(feature_name)
        p2_feature = "player_2_{0}".format(feature_name)
        df = raw_data[["player_1_name", "player_2_name", p1_feature, p2_feature]]

        new_feature_data = np.array([])
        for row in df.itertuples():
            p1 = row.player_1_name
            p2 = row.player_2_name

            p1_opponents = set()
            p1_df_1 = df[(df["player_1_name"] == p1) & (df["player_2_name"] != p2)]
            p1_df_2 = df[(df["player_1_name"] != p2) & (df["player_2_name"] == p1)]

            # Add all of p1's opponents to the set
            p1_opponents.update(list(p1_df_1["player_2_name"].unique()))
            p1_opponents.update(list(p1_df_2["player_1_name"].unique()))

            # p2 opponents
            p2_opponents = set()
            p2_df_1 = df[(df["player_1_name"] == p2) & (df["player_2_name"] != p1)]
            p2_df_2 = df[(df["player_1_name"] != p1) & (df["player_2_name"] == p2)]

            # Add all of p2's opponents to the set
            p2_opponents.update(list(p2_df_1["player_2_name"].unique()))
            p2_opponents.update(list(p2_df_2["player_1_name"].unique()))

            common_opponents = p1_opponents.intersection(p2_opponents)

            # For each player in the common_opponent set, calculate the average feature value of the player_1 matches
            p1_averages = np.array([])
            p2_averages = np.array([])
            for player in common_opponents:
                # This gets the values of the statistic when player_1's name is in the player_1_name column
                as_player_1 = p1_df_1[p1_df_1["player_2_name"] == player][p1_feature].dropna(axis=0, how='any').values
                # This gets the values of the statistic when player_1's name is in the player_2_name column
                as_player_2 = p1_df_2[p1_df_2["player_1_name"] == player][p2_feature].dropna(axis=0, how='any').values

                p1_total_data = np.concatenate((as_player_1, as_player_2), axis=0)

                p1_averages = np.append(p1_averages, p1_total_data.mean())

                # Repeat for player 2
                # This gets the values of the statistic when player_1's name is in the player_1_name column
                as_player_1 = p2_df_1[p2_df_1["player_2_name"] == player][p1_feature].dropna(axis=0, how='any').values
                # This gets the values of the statistic when player_1's name is in the player_2_name column
                as_player_2 = p2_df_2[p2_df_2["player_1_name"] == player][p2_feature].dropna(axis=0, how='any').values

                p2_total_data = np.concatenate((as_player_1, as_player_2), axis=0)

                p2_averages = np.append(p2_averages, p2_total_data.mean())

            # We subtract to get a ratio as to how much better or worse player 1 is
            if p1_averages.size == 0 and p2_averages.size == 0:
                # No common opponents
                temp_df = df[(df["player_1_name"] == p1) & (df["player_2_name"] == p2)].reset_index(drop=True)

                temp_np = np.array([temp_df[p1_feature][0], temp_df[p2_feature][0]])
                common_opponent_score = temp_np.mean()
            elif p1_averages.size != 0 and p2_averages.size == 0:
                common_opponent_score = p1_averages.mean()
            elif p1_averages.size == 0 and p2_averages.size != 0:
                common_opponent_score = p2_averages.mean()
            else:
                common_opponent_score = p1_averages.mean() - p2_averages.mean()

            new_feature_data = np.append(new_feature_data, common_opponent_score)

        # Save the feature so we don't have to run this every time
        feature = pd.DataFrame()
        feature[feature_name] = new_feature_data
        feature.to_csv(data_file, encoding="utf8")


        return feature

def generate_age_feature(data_type):
    dev_pred = []
    age_player1 = data_type['player_1_age']
    age_player2 = data_type['player_2_age']
    for i in range(len(age_player1)):
        if age_player1[i] <= age_player2[i]:
            dev_pred.append(0)
        else:
            dev_pred.append(1)
    age_frame = pd.DataFrame(dev_pred, columns=["age"])
    return age_frame



def generate_avg_rank_feature(dev_raw,players_rank_history,players_match_history):
  dev_pred = []
  player1_train = dev_raw['player_1_name']
  player2_train = dev_raw['player_2_name']
  for i in range(len(player1_train)):
    player1 = player1_train[i]
    player2 = player2_train[i]
    if players_rank_history[player1]/players_match_history[player1] <= players_rank_history[player2]/players_match_history[player2]:
      dev_pred.append(1)
    else:
      dev_pred.append(0)
  avg_rank_frame = pd.DataFrame(dev_pred, columns=["avg_rank"])
  return avg_rank_frame


def generate_win_ratio_feature(dev_raw,train_player_hist,players_match_history):
  dev_pred = []
  player1_train = dev_raw['player_1_name']
  player2_train = dev_raw['player_2_name']
  for i in range(len(player1_train)):
    player1 = player1_train[i]
    player2 = player2_train[i]
    if train_player_hist[player1]/players_match_history[player1] <= train_player_hist[player2]/players_match_history[player2]:
      dev_pred.append(0)
    else:
      dev_pred.append(1)
  win_ratio_frame = pd.DataFrame(dev_pred, columns=["win_ratio"])
  return win_ratio_frame


def generate_rank(player1_ranks, player2_ranks):
    ranks = []
    for i in range(len(player1_ranks)):
        if player1_ranks[i] <= player2_ranks[i]:
            ranks.append(1)
        else:
            ranks.append(0)
    return pd.DataFrame(ranks)


def generate_seed(player1_seed, player2_seed):
    seed = []
    for i in range(len(player1_seed)):
        if math.isnan(float(player1_seed[i])):
            seed1 = 0
        else:
            seed1 = player1_seed[i]
        if math.isnan(float(player2_seed[i])):
            seed2 = 0
        else:
            seed2 = player2_seed[i]
        if seed1 >= seed2:
            seed.append(1)
        else:
            seed.append(0)
    return pd.DataFrame(seed)

def generate_features_and_labels(data_type):
    # data_type is one of: ["train", "validation", "test"]

    data_path = ""
    if data_type == "train":
        data_path = "../data/splits/train.csv"
    elif data_type == "valid":
        data_path = "../data/splits/valid.csv"
    elif data_type == "test":
        data_path = "../data/splits/test.csv"
    else:
        return ""

    # Load the data
    raw_data = pd.read_csv(data_path, encoding="utf8")

    # Create empty data frame for the features
    features = pd.DataFrame()

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

    general_features = [
        "set_score"
    ]

    players = ["player_1", "player_2"]

    # Get the feature corresponding to the set score
    set_score_feature = generate_set_score_feature(raw_data["score"])

    for c in list(set_score_feature):
        features[c] = set_score_feature[c]

    # Get the feature corresponding to the surface
    surface_feature = generate_surface_feature((raw_data["surface"]))
    for c in list(surface_feature):
        features[c] = surface_feature[c]

    # Head to Head feature
    h2h_train = generate_head2head(raw_data[["player_1_name", "player_2_name", "player_1_win"]], data_type)
    features["h2h"] = h2h_train

    # age feature
    age_feature=generate_age_feature(raw_data)
    features["age"]=age_feature

    #average rank feature
    avg_rank_feature=generate_avg_rank_feature(raw_data,players_rank_history,players_match_history)
    features["avg_rank"]=avg_rank_feature

    #win ratio feature
    win_ratio_feature=generate_win_ratio_feature(raw_data,train_player_hist,players_match_history)
    features["win_ratio"]=win_ratio_feature

    ranks_feat = generate_rank(
        raw_data["player_1_rank"], raw_data["player_2_rank"])
    features["ranks"] = ranks_feat

    seeds_feat = generate_seed(
        raw_data["player_1_seed"], raw_data["player_2_seed"])
    features["seeds"] = seeds_feat

    for f in player_specific_features:
        df = pd.read_csv('../data/feature_files/{0}/{0}_{1}.csv'.format(f, data_type))
        features[f] = df[f]

    # Derived features
    # features["completeness"] = math.log(features["1stWon"] + features["2ndWon"]) + math.log(features["bpSaved"]) - math.log(features["bpFaced"])
    features["completeness"] = features["1stWon"] * features["2ndWon"] * features["bpSaved"] / features["bpFaced"]

    # Use this for now to build models
    features["player_1_win"] = raw_data["player_1_win"] # convenience to get the labels to match up correctly
    features = features.dropna(axis=0, how='any')
    features = features[~features.isin([np.nan, np.inf, -np.inf]).any(1)]

    labels = features["player_1_win"]
    features = features.drop(["player_1_win"], axis=1) # undoing the convenience since we don't need it after this point

    return features, labels

if __name__ == "__main__":

    # data_types = ["train", "valid", "test"]
    #
    # for dt in data_types:
    #     data_path = "../data/splits/{0}.csv".format(dt)
    #
    #     # Load the data
    #     raw_data = pd.read_csv(data_path, encoding="utf8")
    #
    #     # Create empty data frame for the features
    #     features = pd.DataFrame()
    #
    #     players = ["player_1", "player_2"]
    #
    #     # Head to Head feature
    #     h2h_train = generate_head2head(raw_data[["player_1_name", "player_2_name", "player_1_win"]], dt)
    #
    #     age_feature=generate_age_feature(raw_data)
    #
    #     avg_rank_feature=generate_avg_rank_feature(raw_data,players_rank_history,players_match_history)
    #
    #     win_ratio_feature = generate_win_ratio_feature(raw_data, train_player_hist, players_match_history)


    generate_features_and_labels("train")

