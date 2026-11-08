# Gamma-Ray vs Hadron Classification (MAGIC Telescope)

Classifies high-energy particle events from the **MAGIC Gamma Telescope** as valuable **gamma-ray signals** (`g`) or **hadronic background noise** (`h`), the kind of filtering the Cherenkov Telescope Array needs to raise its scientific return.

*MSc Business Project in Big Data & AI (M516), Farid Rahimli*

## Results
| Model | Accuracy | AUC | F1 (gamma) | Recall (gamma) |
|---|---|---|---|---|
| **XGBoost (tuned)** | **86.6%** | **0.838** | **0.90** | 0.93 |
| Keras MLP | 83.2% | 0.779 | 0.88 | 0.96 |

XGBoost, tuned with `RandomizedSearchCV`, beat the deep-learning model on this tabular problem.

## Approach
1. Fetch the data from the UCI repository and preprocess it.
2. EDA: feature correlations and distributions; class imbalance (hadrons are ~30% fewer than gammas).
3. Train and tune an XGBoost baseline.
4. Train a Keras MLP.
5. Compare, evaluate and conclude.

## Data
[MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope), UCI Machine Learning Repository. It is fetched in code:
```python
from ucimlrepo import fetch_ucirepo

magic = fetch_ucirepo(id=159)
X, y = magic.data.features, magic.data.targets
```

## Run it
```bash
git clone https://github.com/EFaridRahimli/M516-Submission
cd M516-Submission
conda create -n gamma python=3.10 && conda activate gamma
conda install -c conda-forge tensorflow   # recommended over pip for TensorFlow
pip install -r requirements.txt
jupyter notebook gamma.ipynb
```

## Repository contents
| Path | Description |
|------|-------------|
| `gamma.ipynb` | Full notebook: EDA, models, evaluation |
| `figs/` | Figures used in the report |
| `requirements.txt` | Python dependencies |

## Tech stack
Python · XGBoost · TensorFlow/Keras · scikit-learn · pandas · ucimlrepo
