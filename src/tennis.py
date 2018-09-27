from feature_extractor import generate_features_and_labels
import math
from pprint import pprint
import numpy as np
from time import time

from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_recall_curve, average_precision_score

import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

if __name__ == "__main__":

    ######################
    ## Load Features #####
    ######################

    start = time()

    train_features, train_labels = generate_features_and_labels("train")
    valid_features, valid_labels = generate_features_and_labels("valid")
    test_features, test_labels = generate_features_and_labels("test")


    ######################
    ## SVM ###############
    ######################

    kernels = ["linear", 'poly', 'rbf']
    c_vals = [.25, .5, 1.0, 10]

    # kernels = ["linear"]
    # c_vals = [.25]

    data = {
        kernel: {
            c: {
                "train_accuracy": [],
                "valid_accuracy": [],
                "test_accuracy": []
            } for c in c_vals
        } for kernel in kernels
    }

    percentages = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
    num_examples = train_features.shape[0]

    for k in kernels:
        for c_val in c_vals:
            clf = SVC(kernel=k, C=c_val)
            print("Kernel = {0}, C = {1}".format(k, c_val))
            for p in percentages:
                print("\tp = {0}".format(p))
                size = math.floor(p*num_examples)
                train_examples_subset = train_features[0:size]
                train_label_subset = train_labels[0:size]

                # print("Percentage: {0}, Size: {1}".format(p, size))
                clf.fit(train_examples_subset, train_label_subset)
                training_accuracy = clf.score(train_examples_subset, train_label_subset)
                test_accuracy = clf.score(test_features, test_labels)

                data[k][c_val]["train_accuracy"].append(training_accuracy)
                data[k][c_val]["test_accuracy"].append(test_accuracy)

            # print("Accuracy for {0} kernel, c_val = {1}: {2}".format(k, c_val, clf.score(test_features, test_labels)))

    with open("../results/svm/accuracies.txt", 'wt') as out:
        pprint(data, stream=out)

    # Find the best model based on test accuracy
    print("Finding best model")
    best_model = None
    best_test_accuracy = 0.0
    for k in kernels:
        for c in c_vals:
            if data[k][c]["test_accuracy"][-1] > best_test_accuracy:
                best_test_accuracy = data[k][c]["test_accuracy"][-1]
                best_model = (k, c)

    print("Graphing")
    # Graph the training and test error on the same graph
    best_kernel, best_c = best_model[0], best_model[1]
    training_accuracies = data[best_kernel][best_c]["train_accuracy"]
    test_accuracies = data[best_kernel][best_c]["test_accuracy"]
    # best_kernel = 'linear'
    # best_c = 0.5
    # test_accuracies = [
    #     0.871222076215506,
    #     0.8692509855453351,
    #     0.8575558475689882,
    #     0.8561103810775296,
    #     0.8591327201051249,
    #     0.8629434954007884,
    #     0.8710906701708279,
    #     0.8739816031537451,
    #     0.8718791064388962,
    #     0.871222076215506
    # ]
    #
    # training_accuracies = [
    #     0.8267419962335216,
    #     0.8308754314402259,
    #     0.8245137000627484,
    #     0.8276078431372549,
    #     0.8272054210063998,
    #     0.8293244091194311,
    #     0.8332735747579778,
    #     0.8323137254901961,
    #     0.8315672058003346,
    #     0.832162128246957
    # ]

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(percentages, training_accuracies, linestyle='-', marker='o', color='b', label="Training")
    ax.plot(percentages, test_accuracies, linestyle='-', marker='o', color='g', label="Test")


    plt.title('SVM Accuracies with {0} kernel and C = {1}'.format(best_kernel.upper(), best_c))
    plt.xlabel('Percent of Training Data')
    plt.ylabel('Accuracy')
    ax.legend(loc='lower right')

    plt.savefig("../results/svm/best_model_graph")

    # Cross Validation

    scoring = ['accuracy', 'f1', 'precision', 'recall']
    clf = SVC(kernel=best_kernel, C=best_c) # values for the best model

    cv_results = cross_validate(clf, valid_features, valid_labels, scoring=scoring, cv=5, return_train_score=False)

    with open("../results/svm/5cross_val.txt", 'wt') as out:
        pprint(cv_results, stream=out)

    # Precision-Recall curve
    clf.fit(train_features, train_labels)
    predictions = clf.decision_function(test_features)

    average_precision = average_precision_score(test_labels, predictions)

    precision, recall, _ = precision_recall_curve(test_labels, predictions)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))

    plt.savefig("../results/svm/precisionRecallCurve")

    # Ablation Study
    print("Ablation Study")

    feature_list = [
        "ace",
        "df",
        "svpt",
        "1stIn",
        "1stWon",
        "2ndWon",
        "SvGms",
        "bpFaced",
        "bpSaved",
        "set_score",
        "h2h",
        "completeness",
        "surface",
        "age",
        "avg_rank",
        "win_ratio",
        "ranks",
        "seeds"
    ]

    r = open("../results/ablation_study.txt", 'w')

    ablation_clf = SVC(kernel=best_kernel, C=best_c)
    ablation_clf.fit(train_features, train_labels)

    r.write("Full Set:\n")
    r.write("\tTrain Accuracy: {0}\n".format(ablation_clf.score(train_features, train_labels)))
    r.write("\tValid Accuracy: {0}\n".format(ablation_clf.score(valid_features, valid_labels)))
    r.write("\tTest Accuracy: {0}\n\n".format(ablation_clf.score(test_features, test_labels)))


    full_score = ablation_clf.score(test_features, test_labels)
    print("Full Set done")
    feature_vs_change = []

    for f in feature_list:
        print("Working on feature {0}".format(f))
        ablation_clf = SVC()

        if f == "set_score":
            set_score_features = [
                "straight_bo3",
                "full_bo3",
                "straight_bo5",
                "four_bo5",
                "full_bo5",
                "retirement",
                "walkover"
            ]
            mod_train_features = train_features.drop(set_score_features, axis=1)
            mod_valid_features = valid_features.drop(set_score_features, axis=1)
            mod_test_features = test_features.drop(set_score_features, axis=1)
        elif f == "surface":
            surface_features = [
                "clay",
                "hard",
                "grass",
                "carpet"
            ]
            mod_train_features = train_features.drop(surface_features, axis=1)
            mod_valid_features = valid_features.drop(surface_features, axis=1)
            mod_test_features = test_features.drop(surface_features, axis=1)

        else:
            mod_train_features = train_features.drop([f], axis=1)
            mod_valid_features = valid_features.drop([f], axis=1)
            mod_test_features = test_features.drop([f], axis=1)

        ablation_clf.fit(mod_train_features, train_labels)

        test_accuracy_change = full_score - ablation_clf.score(mod_test_features, test_labels)

        r.write("{0}:\n".format(f))
        r.write("\tTrain Accuracy: {0}\n".format(full_score - ablation_clf.score(mod_train_features, train_labels)))
        r.write("\tValid Accuracy: {0}\n".format(full_score - ablation_clf.score(mod_valid_features, valid_labels)))
        r.write("\tTest Accuracy: {0}\n\n".format(test_accuracy_change))

        feature_vs_change.append((f, test_accuracy_change))

    r.close()


    feature_vs_change = sorted(feature_vs_change, key=lambda x: x[1])

    labels = [x[0] for x in feature_vs_change]
    values = [x[1] for x in feature_vs_change]

    # Create the graph
    fig = plt.figure()
    ax = plt.subplot(111)

    y_pos = np.arange(len(labels))
    ax.bar(y_pos, values, align='center')
    plt.xticks(y_pos, labels)
    plt.ylabel('Percent Contribution')
    plt.xlabel('Features')
    plt.title('Ablation Study')

    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    ax.grid(True)

    plt.savefig("../results/ablation_study")

    end = time()

    print("Time elapsed: {0} minutes".format((end-start)/60))

    svm_training_accuracies = training_accuracies
    svm_test_accuracies = test_accuracies
    mlp_training_accuracies = []
    mlp_test_accuracies = []
    lr_training_accuracies = []
    lr_test_accuracies = []
    for p in percentages:
        size = math.floor(p*num_examples)
        train_examples_subset = train_features[0:size]
        train_label_subset = train_labels[0:size]

        print("MLP with p={0}".format(p))
        clf = MLPClassifier(activation="relu", solver="adam")
        clf.fit(train_examples_subset, train_label_subset)
        mlp_training_accuracies.append(clf.score(train_examples_subset, train_label_subset))
        mlp_test_accuracies.append(clf.score(test_features, test_labels))


        print("LR with p={0}".format(p))
        clf = LogisticRegression()
        clf.fit(train_examples_subset, train_label_subset)
        lr_training_accuracies.append(clf.score(train_examples_subset, train_label_subset))
        lr_test_accuracies.append(clf.score(test_features, test_labels))

    classifiers = ["svm", "mlp", "lr"]
    data_types = ["train", "test"]
    results = {c: {d: None for d in data_types} for c in classifiers}

    results["svm"]["train"] = svm_training_accuracies
    results["svm"]["test"] = svm_test_accuracies
    results["mlp"]["train"] = mlp_training_accuracies
    results["mlp"]["test"] = mlp_test_accuracies
    results["lr"]["train"] = lr_training_accuracies
    results["lr"]["test"] = lr_test_accuracies

    with open("../results/model_comparison.txt", 'wt') as out:
        pprint(results, stream=out)


    ######################
    ## MLP ###############
    ######################
    print("Training and predicting with MLP...")
    clf = MLPClassifier(activation="relu", solver="adam")
    clf.fit(train_features, train_labels)

    dev_accu = clf.score(valid_features, valid_labels)
    print("Dev accuracy: ", dev_accu)

    test_accu = clf.score(test_features, test_labels)
    print("Test accuracy: ", test_accu)

    ######################
    ## Model 3 ###########
    ######################
    print("Training and predicting with Logistic Regression...")
    clf=LogisticRegression()
    clf.fit(train_features, train_labels)

    dev_accu = clf.score(valid_features, valid_labels)
    print("Dev accuracy: ", dev_accu)

    test_accu = clf.score(test_features, test_labels)
    print("Test accuracy: ", test_accu)

    ######################
    ## Evaluation ########
    ######################

