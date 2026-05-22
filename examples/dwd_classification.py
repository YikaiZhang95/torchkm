import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torchkm.estimators import TorchKMDWD


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)

    X, y = make_classification(
        n_samples=120,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        random_state=0,
    )
    X = StandardScaler().fit_transform(X)
    y = np.where(y == 0, -1, 1)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    Cs = np.logspace(2, -2, num=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clf = TorchKMDWD(
        kernel="rbf",
        Cs=Cs,
        nC=len(Cs),
        cv=5,
        device=device,
        random_state=0,
        max_iter=40,
    )
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)
    print("device:", device)
    print("best C:", clf.best_C_)
    print("test accuracy:", float((pred == yte).mean()))


if __name__ == "__main__":
    main()
