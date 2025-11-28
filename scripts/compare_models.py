import json
import matplotlib.pyplot as plt
import os

os.makedirs("visualizations", exist_ok=True)

try:
    with open("../results/pytorch_results.json") as f:
        pytorch = json.load(f)
except FileNotFoundError:
    print("Error: Could not find pytorch_results.json")
    print("Make sure you've run train_pytorch.py first!")
    exit(1)

try:
    with open("../results/sklearn_results.json") as f:
        sklearn = json.load(f)
except FileNotFoundError:
    print("Error: Could not find sklearn_results.json")
    print("Make sure you've run train_sklearn.py first!")
    exit(1)

print("=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)
print(f"\nPyTorch Neural Network:")
print(f"  Accuracy: {pytorch['accuracy']:.4f}")
print(f"  Test Loss: {pytorch['test_loss']:.4f}")
print(f"  Architecture: {pytorch['architecture']}")

print(f"\nScikit-Learn Random Forest:")
print(f"  Accuracy: {sklearn['accuracy']:.4f}")

print("\n" + "=" * 60)
print(f"Winner: {'Scikit-Learn' if sklearn['accuracy'] > pytorch['accuracy'] else 'PyTorch'}")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 6))
models = ["PyTorch", "Scikit-Learn"]
accuracies = [pytorch['accuracy'], sklearn['accuracy']]
colors = ['#FF6B6B', '#4ECDC4']

bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([0.90, 1.0])

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2%}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("visualizations/model_comparison.png", dpi=300)
plt.show()


