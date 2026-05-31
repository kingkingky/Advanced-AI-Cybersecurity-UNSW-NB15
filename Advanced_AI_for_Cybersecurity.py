import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_parquet(
    r"C:\Users\kankrit\OneDrive\Desktop\Advanced AI for Cybersecurity\UNSW_NB15_training-set.parquet"
)

# =====================================================
# Feature Engineering
# =====================================================

df["bytes_ratio"] = df["sbytes"] / (df["dbytes"] + 1)

df["packet_ratio"] = df["spkts"] / (df["dpkts"] + 1)

# =====================================================
# Features / Target
# =====================================================

X = df.drop(columns=["label", "attack_cat"])

y = df["attack_cat"]

# =====================================================
# Columns
# =====================================================

cat_cols = ["proto", "service", "state"]

num_cols = [
    col
    for col in X.columns
    if col not in cat_cols
]

# =====================================================
# Preprocessing
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            cat_cols
        ),
        (
            "num",
            "passthrough",
            num_cols
        )
    ]
)

# =====================================================
# Split
# =====================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# Models
# =====================================================

models = {
    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=3
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        )
}

# =====================================================
# Compare Models
# =====================================================

results = []

best_model_name = None
best_accuracy = 0
best_pipeline = None

for name, model in models.items():

    clf = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    clf.fit(X_train, y_train)

    pred = clf.predict(X_valid)

    acc = accuracy_score(y_valid, pred)

    f1 = f1_score(
        y_valid,
        pred,
        average="weighted"
    )

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1 Score": f1
    })

    print(f"{name}: {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_pipeline = clf

# =====================================================
# Results Table
# =====================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(results_df)

# =====================================================
# Save Model Comparison Plot
# =====================================================

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("images/model_comparison.png")

plt.show()

# =====================================================
# Best Model Evaluation
# =====================================================

print("\nBest Model:", best_model_name)

pred = best_pipeline.predict(X_valid)

print("\nClassification Report")
print(classification_report(y_valid, pred))

# =====================================================
# Confusion Matrix
# =====================================================

ConfusionMatrixDisplay.from_predictions(
    y_valid,
    pred,
    xticks_rotation=90
)

plt.tight_layout()

plt.savefig(
    "images/confusion_matrix.png"
)

plt.show()

# =====================================================
# Cross Validation
# =====================================================

print("\nRunning Cross Validation...")

scores = cross_val_score(
    best_pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("CV Scores:", scores)
print("Mean CV Accuracy:", scores.mean())

# =====================================================
# Feature Importance
# =====================================================

if hasattr(
    best_pipeline.named_steps["model"],
    "feature_importances_"
):

    feature_names = (
        best_pipeline
        .named_steps["preprocessing"]
        .get_feature_names_out()
    )

    importances = (
        best_pipeline
        .named_steps["model"]
        .feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 20 Features")
    print(importance_df.head(20))

    top20 = importance_df.head(20)

    plt.figure(figsize=(10,6))

    plt.barh(
        top20["Feature"],
        top20["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.title(
        f"Top 20 Features ({best_model_name})"
    )

    plt.tight_layout()

    plt.savefig(
        "images/feature_importance.png"
    )

    plt.show()

# =====================================================
# Save Metrics
# =====================================================

with open(
    "results/metrics.txt",
    "w"
) as f:

    f.write(
        f"Best Model: {best_model_name}\n"
    )

    f.write(
        f"Accuracy: {best_accuracy:.4f}\n"
    )

    f.write(
        f"Mean CV Accuracy: {scores.mean():.4f}\n"
    )

# =====================================================
# Save Model
# =====================================================

joblib.dump(
    best_pipeline,
    "best_model.pkl"
)

print("\nModel saved successfully")