import os
import time
import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


N_THREADS = os.cpu_count() or 1
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

from obliquetree import Classifier


class ClassificationSmokeTest(unittest.TestCase):
    def test_obliquetree_vs_sklearn_classifier(self) -> None:
        X, y = make_classification(
            n_samples=100000,
            n_features=24,
            n_informative=16,
            n_redundant=4,
            n_repeated=0,
            n_classes=2,
            class_sep=1.1,
            random_state=0,
        )
        X = X.astype(np.float64, copy=False)
        y = y.astype(np.int64, copy=False)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=0,
            stratify=y,
        )

        oblique_tree = Classifier(
            use_oblique=False,
            max_depth=8,
            min_samples_leaf=4,
            min_samples_split=8,
            random_state=0,
        )
        sklearn_tree = DecisionTreeClassifier(
            criterion="gini",
            max_depth=8,
            min_samples_leaf=4,
            min_samples_split=8,
            random_state=0,
        )

        start = time.perf_counter()
        oblique_tree.fit(X_train, y_train)
        obliquetree_fit_time = time.perf_counter() - start

        start = time.perf_counter()
        obliquetree_pred = oblique_tree.predict(X_test)
        obliquetree_predict_time = time.perf_counter() - start
        obliquetree_accuracy = accuracy_score(y_test, obliquetree_pred)

        start = time.perf_counter()
        sklearn_tree.fit(X_train, y_train)
        sklearn_fit_time = time.perf_counter() - start

        start = time.perf_counter()
        sklearn_pred = sklearn_tree.predict(X_test)
        sklearn_predict_time = time.perf_counter() - start
        sklearn_accuracy = accuracy_score(y_test, sklearn_pred)

        print(
            "\n"
            f"OMP_NUM_THREADS={N_THREADS}\n"
            f"obliquetree fit={obliquetree_fit_time:.6f}s "
            f"predict={obliquetree_predict_time:.6f}s "
            f"accuracy={obliquetree_accuracy:.4f}\n"
            f"sklearn     fit={sklearn_fit_time:.6f}s "
            f"predict={sklearn_predict_time:.6f}s "
            f"accuracy={sklearn_accuracy:.4f}"
        )

        self.assertEqual(obliquetree_pred.shape, y_test.shape)
        self.assertGreaterEqual(obliquetree_accuracy, sklearn_accuracy - 0.05)
        self.assertGreaterEqual(obliquetree_accuracy, 0.75)


if __name__ == "__main__":
    unittest.main()
