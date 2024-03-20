import numpy as np
import matplotlib.pyplot as plt
from DataLoader import DataLoader
from sklearn.inspection import permutation_importance

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    GridSearchCV,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler


# ################## DEFINING DATASET ##################

data_path = "../data/credit.data"
dataset, labels = DataLoader.load_credit_with_onehot(data_path)

################## DEFINING CROSS VALIDATION ##################
outer_cross_validation = RepeatedStratifiedKFold(
    n_splits=3, n_repeats=5, random_state=np.random.randint(1, 1000)
)
inner_cross_validation = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=5, random_state=np.random.randint(1, 1000)
)

################## DEFINING HYPERPARAMETER GRID ##################
knn_parameter_grid = {
    "knn__metric": ["cosine", "euclidean"],
    "knn__n_neighbors": [2, 3, 4],
}

svm_parameter_grid = {
    "svm__C": [0.1, 0.5, 1],
    "svm__kernel": ["poly", "rbf"],
    "svm__degree": [10],
}

decisoin_tree_parameter_grid = {
    "decision_tree__criterion": ["gini", "entropy"],
}

random_forest_parameter_grid = {
    "random_forest__criterion": ["gini", "entropy"],
}


################## OVERALL PERFORMANCE TRACKERS ##################

knn_overall_performance_fscore = []
knn_overall_performance_accuracy = []

svm_overall_performance_fscore = []
svm_overall_performance_accuracy = []

decision_tree_overall_performance_fscore = []
decision_tree_overall_performance_accuracy = []

random_forest_overall_performance_fscore = []
random_forest_overall_performance_accuracy = []

################## PREFORMING NESTED CROSS VALIDATION FOR ACCURACY ##################

print(
    "==================PERFORMING NESTED CROSS VALIDATION FOR ACCURACY=================="
)
counter = 1
for outer_train_indices, outer_test_indices in outer_cross_validation.split(
    dataset, labels
):
    ################## OUTER TRIAN DATASET AND TEST DATASET ##################
    outer_train_dataset = dataset[outer_train_indices]
    outer_train_label = labels[outer_train_indices]

    outer_test_dataset = dataset[outer_test_indices]
    outer_test_label = labels[outer_test_indices]

    ################## HYPERPRAMETER PERFORMANCE LOGS ##################
    knn_performance_log_accuracy = {}
    svm_performance_log_accuracy = {}
    decision_tree_performance_log_accuracy = {}
    random_forest_performance_log_accuracy = {}

    ################## PERFORMING INNER CROSS VALIDATION ##################
    for inner_train_indices, inner_test_indices in inner_cross_validation.split(
        outer_train_dataset, outer_train_label
    ):
        inner_train_dataset = outer_train_dataset[inner_train_indices]
        inner_train_label = outer_train_label[inner_train_indices]

        inner_test_dataset = outer_train_dataset[inner_test_indices]
        inner_test_label = outer_train_label[inner_test_indices]

        ##################  PREPROCESSING ##################
        inner_scalar = MinMaxScaler((-1, 1))
        scaled_inner_train_dataset = inner_scalar.fit_transform(inner_train_dataset)
        scaled_inner_test_dataset = inner_scalar.transform(inner_test_dataset)

        ################## PERFORMING GRID SEARCH ##################

        ################## KNN ##################
        for n_neighbor in knn_parameter_grid["knn__n_neighbors"]:
            for metric in knn_parameter_grid["knn__metric"]:
                knn = KNeighborsClassifier(metric=metric, n_neighbors=n_neighbor)
                knn.fit(scaled_inner_train_dataset, inner_train_label)
                predicted = knn.predict(scaled_inner_test_dataset)

                if (metric, n_neighbor) not in knn_performance_log_accuracy:
                    knn_performance_log_accuracy[(metric, n_neighbor)] = []

                knn_performance_log_accuracy[(metric, n_neighbor)].append(
                    accuracy_score(inner_test_label, predicted)
                )
        ################## SVM ##################
        for C in svm_parameter_grid["svm__C"]:
            for kernel in svm_parameter_grid["svm__kernel"]:
                svm = SVC(C=C, kernel=kernel)
                svm.fit(scaled_inner_train_dataset, inner_train_label)
                predicted = svm.predict(scaled_inner_test_dataset)

                if (C, kernel) not in svm_performance_log_accuracy:
                    svm_performance_log_accuracy[(C, kernel)] = []
                svm_performance_log_accuracy[(C, kernel)].append(
                    accuracy_score(inner_test_label, predicted)
                )

        ################## DECISION TREE ##################
        for criterion in decisoin_tree_parameter_grid["decision_tree__criterion"]:
            decision_tree = DecisionTreeClassifier(criterion=criterion)  # type: ignore

            decision_tree.fit(scaled_inner_train_dataset, inner_train_label)
            predicted = decision_tree.predict(scaled_inner_test_dataset)

            if criterion not in decision_tree_performance_log_accuracy:
                decision_tree_performance_log_accuracy[criterion] = []

            decision_tree_performance_log_accuracy[criterion].append(
                accuracy_score(inner_test_label, predicted)
            )

        ################## RANDOM FOREST ##################
        for criterion in random_forest_parameter_grid["random_forest__criterion"]:
            hyperparameter_accuracy_scores = []
            for i in range(5):
                random_forest = RandomForestClassifier(criterion=criterion)  # type: ignore
                random_forest.fit(scaled_inner_train_dataset, inner_train_label)
                predicted = random_forest.predict(scaled_inner_test_dataset)
                hyperparameter_accuracy_scores.append(
                    accuracy_score(inner_test_label, predicted)
                )

            if criterion not in random_forest_performance_log_accuracy:
                random_forest_performance_log_accuracy[criterion] = []
            random_forest_performance_log_accuracy[criterion].append(
                np.mean(hyperparameter_accuracy_scores)
            )

    ################## FINDING BEST HYPERPRAMETER AND TRAINING ON OUTER TRAIN DATASET AND TEST ON OUTER TEST DATASET ##################

    outer_scalar = MinMaxScaler((-1, 1))
    scaled_outer_train_dataset = outer_scalar.fit_transform(outer_train_dataset)
    scaled_outer_test_dataset = outer_scalar.transform(outer_test_dataset)

    ################## KNN ##################

    best_parameter_knn_accuracy = None
    best_score_knn_accuracy = -float("inf")

    for param_config in knn_performance_log_accuracy:
        v = np.mean(knn_performance_log_accuracy[param_config])
        if v > best_score_knn_accuracy:
            best_score_knn_accuracy = v
            best_parameter_knn_accuracy = param_config

    best_knn_with_respect_to_accuracy = KNeighborsClassifier(
        metric=best_parameter_knn_accuracy[0],  # type: ignore
        n_neighbors=best_parameter_knn_accuracy[1],  # type: ignore
    )

    best_knn_with_respect_to_accuracy.fit(scaled_outer_train_dataset, outer_train_label)
    predicted = best_knn_with_respect_to_accuracy.predict(scaled_outer_test_dataset)
    best_hyperprameter_knn_accuracy = accuracy_score(outer_test_label, predicted)
    knn_overall_performance_accuracy.append(best_hyperprameter_knn_accuracy)

    print("================KNN====================")
    print(f"Best KNN Accuracy param {best_parameter_knn_accuracy}")
    print(f"Best KNN Accuracy {best_hyperprameter_knn_accuracy}")
    print(f"Best KNN Accuracy std {np.std(knn_performance_log_accuracy[(best_parameter_knn_accuracy[0], best_parameter_knn_accuracy[1])])}")  # type: ignore

    confidenc_interval = (
        1.96
        * float(np.std(knn_performance_log_accuracy[(best_parameter_knn_accuracy[0], best_parameter_knn_accuracy[1])]))  # type: ignore
        / float(np.sqrt(len(knn_performance_log_accuracy[(best_parameter_knn_accuracy[0], best_parameter_knn_accuracy[1])])))  # type: ignore
    )

    print(f"Best KNN Accuracy confidence interval {confidenc_interval}")

    ################## SVM ##################
    best_parameter_svm_accuracy = None
    best_score_svm_accuracy = -float("inf")

    for param_config in svm_performance_log_accuracy:
        v = np.mean(svm_performance_log_accuracy[param_config])
        if v > best_score_svm_accuracy:
            best_score_svm_accuracy = v
            best_parameter_svm_accuracy = param_config

    best_svm_with_respect_to_accuracy = SVC(
        C=best_parameter_svm_accuracy[0],  # type: ignore
        kernel=best_parameter_svm_accuracy[1],  # type: ignore
        degree=10,
    )

    best_svm_with_respect_to_accuracy.fit(scaled_outer_train_dataset, outer_train_label)
    predicted = best_svm_with_respect_to_accuracy.predict(scaled_outer_test_dataset)
    best_hyperprameter_svm_accuracy = accuracy_score(outer_test_label, predicted)
    svm_overall_performance_accuracy.append(best_hyperprameter_svm_accuracy)

    print("================SVM====================")
    print(f"Best SVM Accuracy param {best_parameter_svm_accuracy}")
    print(f"Best SVM Accuracy {best_hyperprameter_svm_accuracy}")
    print(f"Best SVM Accuracy std {np.std(svm_performance_log_accuracy[(best_parameter_svm_accuracy[0], best_parameter_svm_accuracy[1])])}")  # type: ignore

    confidenc_interval = (
        1.96
        * float(np.std(svm_performance_log_accuracy[(best_parameter_svm_accuracy[0], best_parameter_svm_accuracy[1])]))  # type: ignore
        / float(np.sqrt(len(svm_performance_log_accuracy[(best_parameter_svm_accuracy[0], best_parameter_svm_accuracy[1])])))  # type: ignore
    )
    print(f"Best SVM Accuracy confidence interval {confidenc_interval}")

    ################## DECISION TREE ##################

    best_parameter_decision_tree_accuracy = None
    best_score_decision_tree_accuracy = -float("inf")

    for param_config in decision_tree_performance_log_accuracy:
        v = np.mean(decision_tree_performance_log_accuracy[param_config])
        if v > best_score_decision_tree_accuracy:
            best_score_decision_tree_accuracy = v
            best_parameter_decision_tree_accuracy = param_config

    best_decision_tree_with_respect_to_accuracy = DecisionTreeClassifier(
        criterion=best_parameter_decision_tree_accuracy  # type: ignore
    )

    best_decision_tree_with_respect_to_accuracy.fit(
        scaled_outer_train_dataset, outer_train_label
    )

    predicted = best_decision_tree_with_respect_to_accuracy.predict(
        scaled_outer_test_dataset
    )

    best_hyperprameter_decision_tree_accuracy = accuracy_score(
        outer_test_label, predicted
    )
    decision_tree_overall_performance_accuracy.append(best_hyperprameter_svm_accuracy)

    print("================DECISION TREE====================")
    print(f"Best Decision Tree Accuracy param {best_parameter_decision_tree_accuracy}")
    print(f"Best Decision Tree Accuracy {best_hyperprameter_decision_tree_accuracy}")
    print(f"Best Decision Tree Accuracy std {np.std(decision_tree_performance_log_accuracy[best_parameter_decision_tree_accuracy])}")  # type: ignore

    confidenc_interval = (
        1.96
        * float(np.std(decision_tree_performance_log_accuracy[best_parameter_decision_tree_accuracy]))  # type: ignore
        / float(np.sqrt(len(decision_tree_performance_log_accuracy[best_parameter_decision_tree_accuracy])))  # type: ignore
    )
    print(f"Best Decision Tree Accuracy confidence interval {confidenc_interval}")

    ################## RANDOM FOREST ##################

    best_parameter_random_forest_accuracy = None
    best_score_random_forest_accuracy = -float("inf")

    for param_config in random_forest_performance_log_accuracy:
        v = np.mean(random_forest_performance_log_accuracy[param_config])
        if v > best_score_random_forest_accuracy:
            best_score_random_forest_accuracy = v
            best_parameter_random_forest_accuracy = param_config

    random_forest_best_hyperparameter_accuracies = []
    for i in range(5):
        best_random_forest_with_respect_to_accuracy = RandomForestClassifier(
            criterion=best_parameter_random_forest_accuracy  # type: ignore
        )

        best_random_forest_with_respect_to_accuracy.fit(
            scaled_outer_train_dataset, outer_train_label
        )

        predicted = best_random_forest_with_respect_to_accuracy.predict(
            scaled_outer_test_dataset
        )

        random_forest_best_hyperparameter_accuracies.append(
            accuracy_score(outer_test_label, predicted)
        )

    best_hyperprameter_random_forest_accuracy = np.mean(
        random_forest_best_hyperparameter_accuracies
    )
    random_forest_overall_performance_accuracy.append(
        best_hyperprameter_random_forest_accuracy
    )

    print("================RANDOM FOREST====================")
    print(f"Best Random Forest Accuracy param {best_parameter_random_forest_accuracy}")
    print(f"Best Random Forest Accuracy {best_hyperprameter_random_forest_accuracy}")
    print(f"Best Random Forest Accuracy std {np.std(random_forest_performance_log_accuracy[best_parameter_random_forest_accuracy])}")  # type: ignore

    confidenc_interval = (
        1.96
        * float(np.std(random_forest_performance_log_accuracy[best_parameter_random_forest_accuracy]))  # type: ignore
        / float(np.sqrt(len(random_forest_performance_log_accuracy[best_parameter_random_forest_accuracy])))  # type: ignore
    )
    print(f"Best Random Forest Accuracy confidence interval {confidenc_interval}")

    print(f"===================={counter}====================")
    counter += 1


################## PREFORMING NESTED CROSS VALIDATION FOR FSCORE ##################
print(
    "=================================================================================================="
)
print(
    "=================================================================================================="
)
print(
    "=================================================================================================="
)
print(
    "==================PERFORMING NESTED CROSS VALIDATION FOR FSCORE=================="
)

counter = 1
for outer_train_indices, outer_test_indices in outer_cross_validation.split(
    dataset, labels
):
    ################## INTER TRIAN DATASET AND TEST DATASET ##################
    outer_train_dataset = dataset[outer_train_indices]
    outer_train_label = labels[outer_train_indices]

    outer_test_dataset = dataset[outer_test_indices]
    outer_test_label = labels[outer_test_indices]

    ################## HYPERPRAMETER PERFORMANCE LOGS ##################
    knn_performance_log_fsocre = {}
    svm_performance_log_fsocre = {}
    decision_tree_performance_log_fsocre = {}
    random_forest_performance_log_fsocre = {}

    ################## PERFORMING INNER CROSS VALIDATION ##################
    for inner_train_indices, inner_test_indices in inner_cross_validation.split(
        outer_train_dataset, outer_train_label
    ):
        inner_train_dataset = outer_train_dataset[inner_train_indices]
        inner_train_label = outer_train_label[inner_train_indices]

        inner_test_dataset = outer_train_dataset[inner_test_indices]
        inner_test_label = outer_train_label[inner_test_indices]

        inner_scalar = MinMaxScaler((-1, 1))
        scaled_inner_train_dataset = inner_scalar.fit_transform(inner_train_dataset)
        scaled_inner_test_dataset = inner_scalar.transform(inner_test_dataset)

        ################## PERFORMING GRID SEARCH ##################

        ################## KNN ##################
        for n_neighbor in knn_parameter_grid["knn__n_neighbors"]:
            for metric in knn_parameter_grid["knn__metric"]:
                knn = KNeighborsClassifier(metric=metric, n_neighbors=n_neighbor)
                knn.fit(scaled_inner_train_dataset, inner_train_label)
                predicted = knn.predict(scaled_inner_test_dataset)

                if (metric, n_neighbor) not in knn_performance_log_fsocre:
                    knn_performance_log_fsocre[(metric, n_neighbor)] = []

                knn_performance_log_fsocre[(metric, n_neighbor)].append(
                    f1_score(inner_test_label, predicted, average="micro")
                )

        ################## SVM ##################
        for C in svm_parameter_grid["svm__C"]:
            for kernel in svm_parameter_grid["svm__kernel"]:
                svm = SVC(C=C, kernel=kernel)
                svm.fit(scaled_inner_train_dataset, inner_train_label)
                predicted = svm.predict(scaled_inner_test_dataset)
                if (C, kernel) not in svm_performance_log_fsocre:
                    svm_performance_log_fsocre[(C, kernel)] = []

                svm_performance_log_fsocre[(C, kernel)].append(
                    f1_score(inner_test_label, predicted, average="micro")
                )

        ################## DECISION TREE ##################
        for criterion in decisoin_tree_parameter_grid["decision_tree__criterion"]:
            decision_tree = DecisionTreeClassifier(criterion=criterion)  # type: ignore

            decision_tree.fit(scaled_inner_train_dataset, inner_train_label)
            predicted = decision_tree.predict(scaled_inner_test_dataset)

            if criterion not in decision_tree_performance_log_fsocre:
                decision_tree_performance_log_fsocre[criterion] = []

            decision_tree_performance_log_fsocre[criterion].append(
                f1_score(inner_test_label, predicted, average="micro")
            )

        ################## RANDOM FOREST ##################
        for criterion in random_forest_parameter_grid["random_forest__criterion"]:
            hyperparameter_f1_scores = []
            for i in range(5):
                random_forest = RandomForestClassifier(criterion=criterion)  # type: ignore
                random_forest.fit(scaled_inner_train_dataset, inner_train_label)
                predicted = random_forest.predict(scaled_inner_test_dataset)

                hyperparameter_f1_scores.append(
                    f1_score(inner_test_label, predicted, average="micro")
                )

            if criterion not in random_forest_performance_log_fsocre:
                random_forest_performance_log_fsocre[criterion] = []

            random_forest_performance_log_fsocre[criterion].append(
                np.mean(hyperparameter_f1_scores)
            )

    ################## FINDING BEST HYPERPRAMETER AND TRAINING ON OUTER TRAIN DATASET AND TEST ON OUTER TEST DATASET ##################

    outer_scalar = MinMaxScaler((-1, 1))
    scaled_outer_train_dataset = outer_scalar.fit_transform(outer_train_dataset)
    scaled_outer_test_dataset = outer_scalar.transform(outer_test_dataset)

    ################## KNN ##################
    best_parameter_knn_fscore = None
    best_score_knn_fscore = -float("inf")

    for param_config in knn_performance_log_fsocre:
        v = np.mean(knn_performance_log_fsocre[param_config])
        if v > best_score_knn_fscore:
            best_score_knn_fscore = v
            best_parameter_knn_fscore = param_config

    best_knn_with_respect_to_fscore = KNeighborsClassifier(
        metric=best_parameter_knn_fscore[0],  # type: ignore
        n_neighbors=best_parameter_knn_fscore[1],  # type: ignore
    )

    best_knn_with_respect_to_fscore.fit(scaled_outer_train_dataset, outer_train_label)

    predicted = best_knn_with_respect_to_fscore.predict(scaled_outer_test_dataset)

    best_hyperprameter_knn_fscore = f1_score(
        outer_test_label, predicted, average="micro"
    )
    knn_overall_performance_fscore.append(best_hyperprameter_knn_fscore)

    print("================KNN====================")
    print(f"Best KNN F1 score param {best_parameter_knn_fscore}")
    print(f"Best KNN F1 score {best_hyperprameter_knn_fscore}")
    print(f"Best KNN F1 score std {np.std(knn_performance_log_fsocre[(best_parameter_knn_fscore[0], best_parameter_knn_fscore[1])])}")  # type: ignore

    confidenc_interval = (
        1.96
        * float(np.std(knn_performance_log_fsocre[(best_parameter_knn_fscore[0], best_parameter_knn_fscore[1])]))  # type: ignore
        / float(np.sqrt(len(knn_performance_log_fsocre[(best_parameter_knn_fscore[0], best_parameter_knn_fscore[1])])))  # type: ignore
    )

    print(f"Best KNN F1 score confidence interval {confidenc_interval}")

    ################## SVM ##################
    best_parameter_svm_fscore = None
    best_score_svm_fscore = -float("inf")

    for param_config in svm_performance_log_fsocre:
        v = np.mean(svm_performance_log_fsocre[param_config])
        if v > best_score_svm_fscore:
            best_score_svm_fscore = v
            best_parameter_svm_fscore = param_config

    best_svm_with_respect_to_fscore = SVC(
        C=best_parameter_svm_fscore[0],  # type: ignore
        kernel=best_parameter_svm_fscore[1],  # type: ignore
        degree=10,
    )

    best_svm_with_respect_to_fscore.fit(scaled_outer_train_dataset, outer_train_label)

    predicted = best_svm_with_respect_to_fscore.predict(scaled_outer_test_dataset)

    best_hyperprameter_svm_fscore = f1_score(
        outer_test_label, predicted, average="micro"
    )

    svm_overall_performance_fscore.append(best_hyperprameter_svm_fscore)

    print("================SVM====================")
    print(f"Best SVM F1 score param {best_parameter_svm_fscore}")
    print(f"Best SVM F1 score {best_hyperprameter_svm_fscore}")

    print(f"Best SVM F1 score std {np.std(svm_performance_log_fsocre[(best_parameter_svm_fscore[0], best_parameter_svm_fscore[1])])}")  # type: ignore

    confidenc_interval = (
        1.96
        * float(np.std(svm_performance_log_fsocre[(best_parameter_svm_fscore[0], best_parameter_svm_fscore[1])]))  # type: ignore
        / float(np.sqrt(len(svm_performance_log_fsocre[(best_parameter_svm_fscore[0], best_parameter_svm_fscore[1])])))  # type: ignore
    )

    print(f"Best SVM F1 score confidence interval {confidenc_interval}")

    ################## DECISION TREE ##################

    best_parameter_decision_tree_fscore = None
    best_score_decision_tree_fscore = -float("inf")

    for param_config in decision_tree_performance_log_fsocre:
        v = np.mean(decision_tree_performance_log_fsocre[param_config])
        if v > best_score_decision_tree_fscore:
            best_score_decision_tree_fscore = v
            best_parameter_decision_tree_fscore = param_config

    best_decision_tree_with_respect_to_fscore = DecisionTreeClassifier(
        criterion=best_parameter_decision_tree_fscore  # type: ignore
    )

    best_decision_tree_with_respect_to_fscore.fit(
        scaled_outer_train_dataset, outer_train_label
    )

    predicted = best_decision_tree_with_respect_to_fscore.predict(
        scaled_outer_test_dataset
    )

    best_hyperprameter_decision_tree_fscore = f1_score(
        outer_test_label, predicted, average="micro"
    )
    decision_tree_overall_performance_fscore.append(best_hyperprameter_svm_fscore)

    print("================DECISION TREE====================")
    print(f"Best Decision Tree F1 score param {best_parameter_decision_tree_fscore}")
    print(f"Best Decision Tree F1 score {best_hyperprameter_decision_tree_fscore}")

    print(f"Best Decision Tree F1 score std {np.std(decision_tree_performance_log_fsocre[best_parameter_decision_tree_fscore])}")  # type: ignore

    confidenc_interval = (
        1.96
        * float(np.std(decision_tree_performance_log_fsocre[best_parameter_decision_tree_fscore]))  # type: ignore
        / float(np.sqrt(len(decision_tree_performance_log_fsocre[best_parameter_decision_tree_fscore])))  # type: ignore
    )

    print(f"Best Decision Tree F1 score confidence interval {confidenc_interval}")

    ################## RANDOM FOREST ##################

    best_parameter_random_forest_fscore = None
    best_score_random_forest_fscore = -float("inf")
    for param_config in random_forest_performance_log_fsocre:
        v = np.mean(random_forest_performance_log_fsocre[param_config])
        if v > best_score_random_forest_fscore:
            best_score_random_forest_fscore = v
            best_parameter_random_forest_fscore = param_config

    random_forest_best_hyperparameter_fscores = []
    for i in range(5):
        best_random_forest_with_respect_to_fscore = RandomForestClassifier(
            criterion=best_parameter_random_forest_fscore  # type: ignore
        )

        best_random_forest_with_respect_to_fscore.fit(
            outer_train_dataset, outer_train_label
        )

        predicted = best_random_forest_with_respect_to_fscore.predict(
            outer_test_dataset
        )

        random_forest_best_hyperparameter_fscores.append(
            f1_score(outer_test_label, predicted, average="micro")
        )

    best_hyperprameter_random_forest_fscore = np.mean(
        random_forest_best_hyperparameter_fscores
    )
    random_forest_overall_performance_fscore.append(
        best_hyperprameter_random_forest_fscore
    )

    print("================RANDOM FOREST====================")
    print(f"Best Random Forest F1 score param {best_parameter_random_forest_fscore}")
    print(f"Best Random Forest F1 score {best_hyperprameter_random_forest_fscore}")

    print(f"Best Random Forest F1 score std {np.std(random_forest_performance_log_fsocre[best_parameter_random_forest_fscore])}")  # type: ignore

    confidenc_interval = (
        1.96
        * float(np.std(random_forest_performance_log_fsocre[best_parameter_random_forest_fscore]))  # type: ignore
        / float(np.sqrt(len(random_forest_performance_log_fsocre[best_parameter_random_forest_fscore])))  # type: ignore
    )

    print(f"Best Random Forest F1 score confidence interval {confidenc_interval}")

    print(f"===================={counter}====================")
    counter += 1


print("================OVERALL====================")

print("=======================KNN=======================")
print(f"KNN F scores {knn_overall_performance_fscore}")
print(f"KNN F scores mean {np.mean(knn_overall_performance_fscore)}")
print(f"KNN F scores std {np.std(knn_overall_performance_fscore)}")
confidenc_interval = (
    1.96
    * float(np.std(knn_overall_performance_fscore))
    / float(np.sqrt(len(knn_overall_performance_fscore)))
)
print(f"KNN F scores confidence interval {confidenc_interval}")


print(f"KNN Accuracy {knn_overall_performance_accuracy}")
print(f"KNN Accuracy mean {np.mean(knn_overall_performance_accuracy)}")
print(f"KNN Accuracy std {np.std(knn_overall_performance_accuracy)}")
confidenc_interval = (
    1.96
    * float(np.std(knn_overall_performance_accuracy))
    / float(np.sqrt(len(knn_overall_performance_accuracy)))
)
print(f"KNN Accuracy confidence interval {confidenc_interval}")


print("=======================SVM=======================")
print(f"SVM F scores {svm_overall_performance_fscore}")
print(f"SVM F scores mean {np.mean(svm_overall_performance_fscore)}")
print(f"SVM F scores std {np.std(svm_overall_performance_fscore)}")
confidenc_interval = (
    1.96
    * float(np.std(svm_overall_performance_fscore))
    / float(np.sqrt(len(svm_overall_performance_fscore)))
)
print(f"SVM F scores confidence interval {confidenc_interval}")

print(f"SVM Accuracy {svm_overall_performance_accuracy}")
print(f"SVM Accuracy mean {np.mean(svm_overall_performance_accuracy)}")
print(f"SVM Accuracy std {np.std(svm_overall_performance_accuracy)}")
confidenc_interval = (
    1.96
    * float(np.std(svm_overall_performance_accuracy))
    / float(np.sqrt(len(svm_overall_performance_accuracy)))
)
print(f"SVM Accuracy confidence interval {confidenc_interval}")

print("=======================DECISION TREE=======================")
print(f"Decision Tree F scores {decision_tree_overall_performance_fscore}")
print(
    f"Decision Tree F scores mean {np.mean(decision_tree_overall_performance_fscore)}"
)
print(f"Decision Tree F scores std {np.std(decision_tree_overall_performance_fscore)}")
confidenc_interval = (
    1.96
    * float(np.std(decision_tree_overall_performance_fscore))
    / float(np.sqrt(len(decision_tree_overall_performance_fscore)))
)
print(f"Decision Tree F scores confidence interval {confidenc_interval}")

print(f"Decision Tree Accuracy {decision_tree_overall_performance_accuracy}")
print(
    f"Decision Tree Accuracy mean {np.mean(decision_tree_overall_performance_accuracy)}"
)
print(
    f"Decision Tree Accuracy std {np.std(decision_tree_overall_performance_accuracy)}"
)
confidenc_interval = (
    1.96
    * float(np.std(decision_tree_overall_performance_accuracy))
    / float(np.sqrt(len(decision_tree_overall_performance_accuracy)))
)
print(f"Decision Tree Accuracy confidence interval {confidenc_interval}")

print("=======================RANDOM FOREST=======================")
print(f"Random Forest F scores {random_forest_overall_performance_fscore}")
print(
    f"Random Forest F scores mean {np.mean(random_forest_overall_performance_fscore)}"
)
print(f"Random Forest F scores std {np.std(random_forest_overall_performance_fscore)}")
confidenc_interval = (
    1.96
    * float(np.std(random_forest_overall_performance_fscore))
    / float(np.sqrt(len(random_forest_overall_performance_fscore)))
)
print(f"Random Forest F scores confidence interval {confidenc_interval}")

print(f"Random Forest Accuracy {random_forest_overall_performance_accuracy}")
print(
    f"Random Forest Accuracy mean {np.mean(random_forest_overall_performance_accuracy)}"
)
print(
    f"Random Forest Accuracy std {np.std(random_forest_overall_performance_accuracy)}"
)
confidenc_interval = (
    1.96
    * float(np.std(random_forest_overall_performance_accuracy))
    / float(np.sqrt(len(random_forest_overall_performance_accuracy)))
)
print(f"Random Forest Accuracy confidence interval {confidenc_interval}")


############################################## DECISON TREE ##############################################

data_path = "../data/credit.data"
dataset, labels = DataLoader.load_credit(data_path)

decsion_tree = DecisionTreeClassifier()
decsion_tree.fit(dataset, labels)

print("Feature importance : ", decsion_tree.feature_importances_)

plot_tree(decsion_tree, max_depth=2, rounded=True, precision=10)
features = []
for i, a in enumerate(decsion_tree.feature_importances_):
    print(i, a)
    features.append((i, a))

sorted_features = sorted(features, key=lambda x: x[1], reverse=True)
print("Sorted features : ", sorted_features)

plt.show()


############################################## SVC ##############################################
data_path = "../data/credit.data"
dataset, labels = DataLoader.load_credit_with_onehot(data_path)

svm = SVC(kernel="linear", C=1)
svm.fit(dataset, labels)

print("suppoer vectors : ", svm.support_vectors_)
print("support vectors shape : ", svm.support_vectors_.shape)
print("support num ", svm.n_support_)

predicted = svm.predict(dataset)
print("accuracy score : ", accuracy_score(labels, predicted))


print("weight_vector: ", svm.coef_)

# support_vectors = svm.support_vectors_
# support_vector_labels = svm.dual_coef_.ravel() > 0
# positive_support_vectors = support_vectors[support_vector_labels]
# negative_support_vectors = support_vectors[~support_vector_labels]

# print("positive support vectors : ", len(positive_support_vectors))
# for positive_support_vector in positive_support_vectors:
#     print(positive_support_vector)

# print("negative support vectors : ", len(negative_support_vectors))
# for negative_support_vector in negative_support_vectors:
#     print(negative_support_vector)


# weight_vector = np.array(
#     [
#         [
#             -3.10269356,
#             -2.76853478,
#             1.43196412,
#             4.43926422,
#             -0.0443037232,
#             -4.45343471,
#             -1.98140393,
#             0.159950899,
#             1.49919274,
#             4.77569501,
#             -2.96173124,
#             5.94428244,
#             0.744161602,
#             1.21888194,
#             -0.38170544,
#             -2.0801592,
#             -2.37714551,
#             0.0,
#             0.0,
#             -0.205750435,
#             0.099165844,
#             -0.00126657682,
#             -2.80589951,
#             -2.19663047,
#             0.169053582,
#             2.17272469,
#             2.6607517,
#             -0.036632247,
#             -0.624154966,
#             -1.00705297,
#             3.0749229,
#             -1.40708272,
#             -2.51606688,
#             -1.83094642,
#             -0.341755759,
#             2.53442366,
#             -0.361721486,
#             0.0,
#             -0.282453815,
#             -1.0,
#             1.28245382,
#             -0.227722267,
#             0.828697577,
#             0.0283383697,
#             0.247872132,
#             -1.10490808,
#             -0.0352913528,
#             -0.627030084,
#             -1.78333108,
#             2.41036117,
#             -1.47710126,
#             0.750390693,
#             0.726710571,
#             0.036428242,
#             -0.82541935,
#             1.24582207,
#             -0.473408651,
#             0.0530059336,
#             0.763903313,
#             -0.460190242,
#             0.460190242,
#             -0.487722547,
#             0.487722547,
#         ]
#     ]
# )
# lst = []
# i = 0
# for x in weight_vector[0]:
#     lst.append((i, x))
#     i += 1

# sorted_lst = sorted(lst, key=lambda x: x[1], reverse=True)

# print("sorted lst : ", sorted_lst)
