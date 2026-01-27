import json
import pandas as pd
from MiniAutoML.automl import MiniAutoML
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # df = pd.read_csv("data/spaceship_titanic.csv")

    # X = df.drop(columns=["Transported"])
    # y = df["Transported"]

    X = pd.read_csv("data/X_adult.csv")
    y = pd.read_csv("data/y_adult.csv")

    with open("data/models_example.json", "r") as f:
        models = json.load(f)

    automl = MiniAutoML(models_config=models)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    automl.fit(X_train, y_train)
    y_pred = automl.predict(X_test)

    print("Balanced Accuracy:", balanced_accuracy_score(
        y_test, y_pred))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    automl.print_best_models()
