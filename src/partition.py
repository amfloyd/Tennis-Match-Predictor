import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    column_headers = None

    raw_data = []

    for i in range(2000, 2018):
        raw_data.append(pd.read_csv('../data/raw/atp_matches_{0}.csv'.format(i)))

    all_data = pd.concat(raw_data)

    # Get rid of the Davis Cup Matches
    all_data = all_data[~all_data['tourney_name'].str.contains('Davis Cup')]

    columns_to_change = []
    for c in all_data.columns:
        if "winner" in c or "loser" in c or c.startswith("w_") or c.startswith("l_"):
            columns_to_change.append(c)

    # Create dictionary of new column names
    rename_mapping = {}
    for c in columns_to_change:
        if c.startswith("winner"):
            rename_mapping[c] = c.replace("winner", "player_1")
        elif c.startswith("loser"):
            rename_mapping[c] = c.replace("loser", "player_2")
        elif c.startswith("w_"):
            rename_mapping[c] = c.replace("w_", "player_1_")
        elif c.startswith("l_"):
            rename_mapping[c] = c.replace("l_", "player_2_")


    all_data.rename(columns=rename_mapping, inplace=True)

    # Swap player_1 data and player_2 data for 50% of the examples
    mask = np.random.choice([False, True], len(all_data), p=[0.5, 0.5])


    # Get the columns to swap
    columns_to_swap = [x.split("player_1_")[1] for x in all_data.columns if "player_1" in x]

    for c in columns_to_swap:

        # Labels to swap
        p1 = "player_1_{0}".format(c)
        p2 = "player_2_{0}".format(c)

        all_data.loc[mask, [p1, p2]] = all_data.loc[mask, [p2, p1]].values

    # Assign the labels according to the mask
    all_data["player_1_win"] = 1
    all_data["player_1_win"][mask] = 0

    # Get the years so we can split the data
    years = all_data["tourney_id"]
    years = pd.Series([y[0:4] for y in years.values])

    # Split into train, validation, and test bydate
    # [2000, 2008] = 50% of data
    # [2009, 2013] = 28% of data
    # [2013, 2017] = 22% of data
    train_shape = years[(years >= '2000') & (years <= '2008')].shape[0]
    valid_shape = years[(years >= '2009') & (years <= '2013')].shape[0]
    test_shape = years[(years >= '2014') & (years <= '2017')].shape[0]

    # Partition
    x_train = all_data[0:train_shape]
    x_valid = all_data[train_shape: train_shape + valid_shape]
    x_test = all_data[train_shape + valid_shape: train_shape + valid_shape + test_shape]

    # Shuffle
    x_train = x_train.sample(frac=1).reset_index(drop=True)
    x_valid = x_valid.sample(frac=1).reset_index(drop=True)
    x_test = x_test.sample(frac=1).reset_index(drop=True)

    # Save the splits
    x_train.to_csv('../data/splits/train.csv', encoding="utf8")
    x_valid.to_csv('../data/splits/valid.csv', encoding="utf8")
    x_test.to_csv('../data/splits/test.csv', encoding="utf8")
