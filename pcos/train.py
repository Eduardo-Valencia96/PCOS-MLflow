from sklearn.linear_model import LogisticRegression


def train_model(
    X_train,
    y_train,
    C=1.0,
    max_iter=1000,
    solver="liblinear",
    random_state=42,
):
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    return model