import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.title("Model Comparison")
plt.ylabel("Accuracy")

plt.tight_layout()

plt.savefig("model_comparison.png")

plt.show()