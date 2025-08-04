import numpy as np
from sklearn.metrics import precision_recall_curve
import pandas as pd

# From the pipeline output:
actual_positives = 12  # Actual non-zero LTV users
total_samples = 26202  # Total test samples
predicted_positives_at_current_threshold = 6319  # Users predicted as non-zero
current_recall = 0.50  # Model achieved 50% recall
current_precision = 0.00  # Precision at current threshold (rounded)

print("=== PRECISION AT 0.9 RECALL ANALYSIS ===")
print(f"Current model performance:")
print(f"- Recall: {current_recall}")
print(f"- Predicted positives: {predicted_positives_at_current_threshold}")
print(f"- Actual positives: {actual_positives}")
print(f"- True positives at current threshold: {int(current_recall * actual_positives)}")

# Calculate what would happen at 0.9 recall
target_recall = 0.9
true_positives_at_90_recall = int(target_recall * actual_positives)

print(f"\nAt 0.9 recall:")
print(f"- True positives needed: {true_positives_at_90_recall}")
print(f"- Remaining false positives: {actual_positives - true_positives_at_90_recall}")

# Estimate precision at 0.9 recall
# Assuming we need to lower threshold significantly to catch 90% of positives
# This would likely increase false positives dramatically
estimated_false_positives_at_90_recall = predicted_positives_at_current_threshold * 2  # Conservative estimate

estimated_precision_at_90_recall = true_positives_at_90_recall / (true_positives_at_90_recall + estimated_false_positives_at_90_recall)

print(f"\nEstimated precision at 0.9 recall:")
print(f"- True positives: {true_positives_at_90_recall}")
print(f"- Estimated false positives: {estimated_false_positives_at_90_recall}")
print(f"- Estimated precision: {estimated_precision_at_90_recall:.4f}")

print(f"\nREALITY CHECK:")
print(f"With only {actual_positives} positive cases out of {total_samples} total samples,")
print(f"achieving high recall while maintaining reasonable precision is extremely challenging.")
print(f"The class imbalance ratio is 1:{total_samples//actual_positives}")