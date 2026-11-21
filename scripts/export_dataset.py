from pathlib import Path
from model import generate_synthetic_dataset

output = Path(__file__).resolve().parents[1] / "data" / "synthetic_housing_dataset.csv"
output.parent.mkdir(parents=True, exist_ok=True)
generate_synthetic_dataset().to_csv(output, index=False)
print(f"Exported reproducible 2,000-row synthetic dataset to {output}")
