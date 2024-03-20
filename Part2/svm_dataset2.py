import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.pipeline import make_pipeline


def compute_confidence_interval(std, n):
    return 1.96 * (std / np.sqrt(n))


dataset, labels = pickle.load(open("../data/part2_dataset2.data", "rb"))
cross_validation = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
parameter_grid = {
    "svc__C": [0.1, 0.5, 1],
    "svc__kernel": ["rbf", "sigmoid", "poly"],
    "svc__degree": [10],
}
svm = SVC()

svb_pipeline = make_pipeline(StandardScaler(), svm)

gridsearch_cross_validation = GridSearchCV(
    svb_pipeline,
    parameter_grid,
    scoring="accuracy",
    cv=cross_validation,  # type: ignore
    verbose=True,
    refit=True,
)

gridsearch_cross_validation.fit(dataset, labels)


hyperparameters = gridsearch_cross_validation.cv_results_["params"]
means = gridsearch_cross_validation.cv_results_["mean_test_score"]
stds = gridsearch_cross_validation.cv_results_["std_test_score"]
n = gridsearch_cross_validation.cv_results_["split0_test_score"].shape[0]
results = zip(means, stds, hyperparameters)

for mean, std, param in results:
    confidence_interval = compute_confidence_interval(std, n)
    print("=====================================")
    print(f"hyperprameters={param}")
    print(
        f"mean={mean}, std={std}, param={param}, confidence_interval(95%)={confidence_interval}"
    )
