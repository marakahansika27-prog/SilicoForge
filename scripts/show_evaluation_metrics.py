import csv
import os

CSV_PATH = r"outputs\hackathon_v3\phase28\phase28_final_results.csv"

errors = []

with open(CSV_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        errors.append(float(row["err_winning"]))

total = len(errors)

success_10 = sum(e <= 10.0 for e in errors)
success_1 = sum(e <= 1.0 for e in errors)

mean_error = sum(errors) / total
sorted_errors = sorted(errors)
median_error = (
    sorted_errors[total // 2]
    if total % 2 == 1
    else (sorted_errors[total // 2 - 1] + sorted_errors[total // 2]) / 2
)

print("=" * 60)
print("          DRIFT-SENSE V3 EVALUATION RESULTS")
print("=" * 60)

print(f"Total Test Cases          : {total}")

print()
print("LOCALIZATION PERFORMANCE")
print("-" * 60)

print(
    f"Success @ <= 10 px        : "
    f"{success_10}/{total} = {success_10 / total * 100:.1f}%"
)

print(
    f"Near-Perfect @ <= 1 px    : "
    f"{success_1}/{total} = {success_1 / total * 100:.1f}%"
)

print()
print("ERROR METRICS")
print("-" * 60)

print(f"Mean Localization Error   : {mean_error:.2f} px")
print(f"Median Localization Error : {median_error:.2f} px")
print(f"Best Localization Error   : {min(errors):.4f} px")
print(f"Worst Localization Error  : {max(errors):.2f} px")

print()
print("=" * 60)
print(f"FINAL LOCALIZATION ACCURACY: {success_10 / total * 100:.1f}%")
print(f"SUCCESSFUL CASES: {success_10}/{total}")
print("=" * 60)