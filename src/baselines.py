import pandas as pd
import numpy as np
import math 
import pickle

def extract_win_cnts(raw_data):
  players_win_history = {}
  player1s = raw_data['player_1_name']
  player2s = raw_data['player_2_name']
  players_wins = raw_data['player_1_win']
  for i in range(len(player1s)):
    player1 = player1s[i]
    player2 = player2s[i]
    if player1 in players_win_history:
      players_win_history[player1] += players_wins[i]
    else:
      players_win_history[player1] = players_wins[i]
    if player2 in players_win_history:
      if players_wins[i] == 0:
        players_win_history[player2] += 1
    else:
      if players_wins[i] == 0:
        players_win_history[player2] = 1
      else:
        players_win_history[player2] = 0
  return players_win_history

def matches_played(raw_data):
  player1s = raw_data['player_1_name']
  player2s = raw_data['player_2_name']
  rank = 0
  players_match_history = {}
  for i in range(len(player1s)):
    player1 = player1s[i]
    player2 = player2s[i]
    if player1 in players_match_history:
      players_match_history[player1] += 1
    else:
      players_match_history[player1] = 1

    if player2 in players_match_history:
      players_match_history[player2] += 1
    else:
      players_match_history[player2] = 1
  return players_match_history

def avg_rank(raw_data):
  player1s = raw_data['player_1_name']
  player2s = raw_data['player_2_name']
  rank=0
  players_rank_history = {}
  for i in range(len(player1s)):
    player1 = player1s[i]
    player2 = player2s[i]
    if math.isnan(raw_data['player_1_rank'][i]):
      raw_data['player_1_rank'][i]=1000
    if player1 in players_rank_history:
      players_rank_history[player1] += raw_data['player_1_rank'][i]
    else:
        players_rank_history[player1] = raw_data['player_1_rank'][i]
    if math.isnan(raw_data['player_2_rank'][i]):
      raw_data['player_2_rank'][i] = 1000
    if player2 in players_rank_history:
      players_rank_history[player2] += raw_data['player_2_rank'][i]
    else:
      players_rank_history[player2] = raw_data['player_2_rank'][i]
  return players_rank_history

def h2h_baseline(raw_data, players_win_hist):
  player1s = np.array(raw_data['player_1_name'])
  player1s = np.reshape(player1s, (len(player1s),1))
  player2s = np.array(raw_data['player_2_name'])
  player2s = np.reshape(player2s, (len(player2s), 1))
  player_pairs = np.hstack((player1s, player2s))

  pred = []
  for player1, player2 in player_pairs:
    if player1 in players_win_hist and player2 in players_win_hist:
      if players_win_hist[player1] >= players_win_hist[player2]:
        player_1_win = 1
      else:
        player_1_win = 0
    elif player1 in players_win_hist and player2 not in players_win_hist:
      player_1_win = 1
    elif player1 not in players_win_hist and player2 in players_win_hist:
      player_1_win = 0
    else:
      player_1_win = 1
    pred.append(player_1_win)
  
  return pred


def rank_baseline(raw):
  pred = []
  ranks1 = raw['player_1_rank']
  ranks2 = raw['player_2_rank']
  for i in range(len(ranks1)):
    if ranks1[i] <= ranks2[i]:
      pred.append(1)
    else:
      pred.append(0)
  return pred

#age baseline- an older, more experienced player will always be pegged to win
def age_baseline(raw):
  pred = []
  age_player1 = raw['player_1_age']
  age_player2 = raw['player_2_age']
  for i in range(len(age_player1)):
    if age_player1[i] <= age_player2[i]:
      pred.append(0)
    else:
      pred.append(1)
  return pred

# def avg_rank_model(dev_raw,test_raw,players_rank_history,players_match_history):
#   dev_pred = []
#   player1_train = dev_raw['player_1_name']
#   player2_train = dev_raw['player_2_name']
#   for i in range(len(player1_train)):
#     player1 = player1_train[i]
#     player2 = player2_train[i]
#     if players_rank_history[player1]/players_match_history[player1] <= players_rank_history[player2]/players_match_history[player2]:
#       dev_pred.append(1)
#     else:
#       dev_pred.append(0)
#   test_pred = []
#   player1_test = dev_raw['player_1_name']
#   player2_test = dev_raw['player_2_name']
#   for i in range(len(player1_test)):
#     player1 = player1_test[i]
#     player2 = player2_test[i]
#     if players_rank_history[player1]/players_match_history[player1] <= players_rank_history[player2]/players_match_history[player2]:
#       test_pred.append(1)
#     else:
#       test_pred.append(0)
#   return dev_pred, test_pred

# def win_ratio_model(dev_raw,test_raw,train_player_hist,players_match_history):
#   dev_pred = []
#   player1_train = dev_raw['player_1_name']
#   player2_train = dev_raw['player_2_name']
#   for i in range(len(player1_train)):
#     player1 = player1_train[i]
#     player2 = player2_train[i]
#     if train_player_hist[player1]/players_match_history[player1] <= train_player_hist[player2]/players_match_history[player2]:
#       dev_pred.append(0)
#     else:
#       dev_pred.append(1)
#   test_pred = []
#   player1_test = dev_raw['player_1_name']
#   player2_test = dev_raw['player_2_name']
#   for i in range(len(player1_test)):
#     player1 = player1_test[i]
#     player2 = player2_test[i]
#     if train_player_hist[player1]/players_match_history[player1] <= train_player_hist[player2]/players_match_history[player2]:
#       test_pred.append(0)
#     else:
#       test_pred.append(1)
#   return dev_pred, test_pred



def seed_baseline(raw):
  pred = []
  seeds1 = raw['player_1_seed']
  seeds2 = raw['player_2_seed']
  for i in range(len(seeds1)):
    if math.isnan(float(seeds1[i])):
      seed1 = 0
    else:
      seed1 = seeds1[i]
    if math.isnan(float(seeds2[i])):
      seed2 = 0
    else:
      seed2 = seeds2[i]
    if seed1 >= seed2:
      pred.append(1)
    else:
      pred.append(0)
  return pred


def evaluate(pred, raw_data):
  gt = raw_data['player_1_win']
  cnt = 0
  for i in range(len(gt)):
    if gt[i] == pred[i]:
      cnt += 1
  accuracy = cnt/len(gt)
  return accuracy

if __name__ ==  "__main__":
  train_file = "../data/splits/train.csv"
  dev_file = "../data/splits/valid.csv"
  test_file = "../data/splits/test.csv"
  # complete_file="../data/splits/all.csv"

  train_raw = pd.read_csv(train_file, encoding="utf8")
  dev_raw = pd.read_csv(dev_file, encoding="utf8")
  test_raw = pd.read_csv(test_file, encoding="utf8")
  # all_raw=pd.read_csv(complete_file, encoding="utf8")
  
  print("Modeling head to head baseline...")
  # print("Extracting player's winning history counts...")
  train_player_hist = extract_win_cnts(train_raw)
  #test_player_hist = extract_win_cnts(test_raw)

  # print("Predicting on dev and test sets...")
  h2h_train_pred = h2h_baseline(train_raw, train_player_hist)
  h2h_dev_pred = h2h_baseline(dev_raw, train_player_hist)
  h2h_test_pred = h2h_baseline(test_raw, train_player_hist)

  h2h_train_accu = evaluate(h2h_train_pred, train_raw)
  h2h_dev_accu = evaluate(h2h_dev_pred, dev_raw)
  h2h_test_accu = evaluate(h2h_test_pred, test_raw)
  print("Head to head baseline train accuracy: ", h2h_train_accu)
  print("Head to head baseline dev accuracy: ", h2h_dev_accu)
  print("Head to head baseline test accuracy: ", h2h_test_accu)

  print("Modeling rank baseline...")
  rank_train_pred = rank_baseline(train_raw)
  rank_dev_pred = rank_baseline(dev_raw)
  rank_test_pred = rank_baseline(test_raw)

  rank_train_accu = evaluate(rank_train_pred, train_raw)
  rank_dev_accu = evaluate(rank_dev_pred, dev_raw)
  rank_test_accu = evaluate(rank_test_pred, test_raw)
  print("Rank baseline train accuracy: ", rank_train_accu)
  print("Rank baseline dev accuracy: ", rank_dev_accu)
  print("Rank baseline test accuracy: ", rank_test_accu)

  print("Modeling seed baseline...")
  seed_train_pred = seed_baseline(train_raw)
  seed_dev_pred = seed_baseline(dev_raw)
  seed_test_pred = seed_baseline(test_raw)

  seed_train_accu = evaluate(seed_train_pred, train_raw)
  seed_dev_accu = evaluate(seed_dev_pred, dev_raw)
  seed_test_accu = evaluate(seed_test_pred, test_raw)
  print("Seed baseline train accuracy: ", seed_train_accu)
  print("Seed baseline dev accuracy: ", seed_dev_accu)
  print("Seed baseline test accuracy: ", seed_test_accu)

  print("Modeling age baseline...")
  age_train_pred = age_baseline(train_raw)
  age_dev_pred = age_baseline(dev_raw)
  age_test_pred = age_baseline(test_raw)

  age_train_accu = evaluate(age_train_pred, train_raw)
  age_dev_accu = evaluate(age_dev_pred, dev_raw)
  age_test_accu = evaluate(age_test_pred, test_raw)
  print("Age baseline train accuracy: ", age_train_accu)
  print("Age baseline dev accuracy: ", age_dev_accu)
  print("Age baseline test accuracy: ", age_test_accu)

  
  # players_rank_history=avg_rank(all_raw)
  # f = open("rank_history.pkl", "wb")
  # pickle.dump(players_rank_history, f)
  # f.close()
  # players_match_history=matches_played(all_raw)
  # f = open("total_match.pkl","wb")
  # pickle.dump(players_match_history,f)
  # f.close()

 
  # print("Modeling average rank ...")
  # age_dev_pred, age_test_pred = avg_rank_model(dev_raw, test_raw, players_rank_history, players_match_history)
  # age_dev_accu = evaluate(age_dev_pred, dev_raw)
  # age_test_accu = evaluate(age_test_pred, test_raw)
  # print("Average rank dev accuracy: ", age_dev_accu)
  # print("Average rank test accuracy: ", age_test_accu)
  #
  # print("Modeling win ratio ...")
  # age_dev_pred, age_test_pred = win_ratio_model(dev_raw,test_raw,train_player_hist,players_match_history)
  # age_dev_accu = evaluate(age_dev_pred, dev_raw)
  # age_test_accu = evaluate(age_test_pred, test_raw)
  # print("Win ratio dev accuracy: ", age_dev_accu)
  # print("Win ratio test accuracy: ", age_test_accu)

