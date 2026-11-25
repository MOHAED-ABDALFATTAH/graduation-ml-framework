# Getting started (read this first)

## What each tool is for

| Tool | Used for |
|---|---|
| **GitHub** | All code and configs. Pull requests happen here. |
| **DagsHub** | The shared MLflow dashboard where every experiment appears. |
| **Google Drive** | Datasets, final reports, important checkpoints. |
| **Google Colab** | Where each of us trains and evaluates models. |

Two tracks run in parallel and merge later into one pipeline:
**Detection** (where are the pests?) and **Classification** (which pest is it?).
Each track has its own MLflow experiment: `Detection`, `Classification`, and later `Pipeline`.

Experiment **results** are never committed to Git. They go to MLflow automatically.
Git holds **code and configs** only.

## One-time setup (about 20 minutes)

1. **Accept the invitations** (check your email): GitHub repo and DagsHub repo.
   - GitHub: https://github.com/MOHAED-ABDALFATTAH/graduation-ml-framework
   - DagsHub: https://dagshub.com/MOHAED-ABDALFATTAH/graduation-ml-framework
2. **Create a DagsHub access token:** DagsHub -> your profile -> Settings -> Tokens -> create one. Copy it.
   Treat it like a password. Never paste it into a notebook, a chat, or Git.
3. **Add three Colab secrets** (Colab -> key icon in the left sidebar). Turn on *Notebook access* for each:

   | Secret name | Value |
   |---|---|
   | `MLFLOW_TRACKING_URI` | `https://dagshub.com/MOHAED-ABDALFATTAH/graduation-ml-framework.mlflow` |
   | `MLFLOW_USERNAME` | your DagsHub username |
   | `MLFLOW_PASSWORD` | your DagsHub token |

4. **Add the shared Drive folder to your Drive:** right-click `Graduation_Project` -> Organize -> Add shortcut -> My Drive.
5. **Run the connection test:** open `notebooks/00_mlflow_connection_test.ipynb` in Colab
   (GitHub -> open the file -> "Open in Colab", or upload it), set `OWNER` to your name, run all cells.
6. **Check DagsHub -> Experiments tab.** You should see your runs under `Connection_Test`.
   Delete them afterwards (select -> Delete).

If something fails:

| Error | Likely cause |
|---|---|
| 401 / 403 | Wrong username or token, or you have not accepted the DagsHub invitation |
| 404 | The URI has a typo. It must end in `.mlflow` with no trailing slash |
| Secret not found | Name misspelled, or *Notebook access* is off |

## Working with Git (Windows)

Clone once:

```powershell
git clone https://github.com/MOHAED-ABDALFATTAH/graduation-ml-framework.git
cd graduation-ml-framework
```

**Nobody pushes directly to `main`.** For every change:

```powershell
git checkout main
git pull origin main
git checkout -b feature/short-description      # or exp/det-yolov8-augmentation
# ... edit files ...
git add .
git status                                     # check that .env or data files are NOT listed
git commit -m "Describe the change"
git push origin feature/short-description
```

Then open GitHub and click **Compare & pull request**. One teammate reviews and merges.

Branch names:
- `feature/<module>` for framework code (e.g. `feature/evaluation`)
- `exp/<track>-<description>` for an experiment (e.g. `exp/det-yolov8-baseline`)

## How to run an experiment

You do **not** need permission to run an experiment. Every run appears in MLflow with its Git commit.

1. Create a branch `exp/<track>-<description>`.
2. If your experiment needs new settings, add a small config file in `configs/`.
3. Run it in Colab from that branch.
4. Open a pull request so the config and code become part of `main`.

Run names follow this pattern: `DET-2026-001_YOLOv8_Baseline`, `CLS-2026-001_ResNet50_Baseline`.
Every run must record: owner, dataset version, seed, Git commit (the helper in `src/tracking/` does this for you).

```python
from src.tracking import setup_tracking, start_tracked_run, log_metrics_safe

setup_tracking("Detection")
with start_tracked_run("DET-2026-001_YOLOv8_Baseline", owner="member_a",
                       track="detection", dataset_version="detection_v1",
                       seed=42, config=cfg_dict):
    ...  # train and evaluate
    log_metrics_safe(metrics)
```

If your code crashes, the run is kept, marked `failed`, and the error is saved as an artifact.

## Rules we all follow

1. The dataset is never edited. A change means a new version (`detection_v2`).
2. The test set is used only for final evaluation, never for tuning.
3. The class order is frozen (see `docs/protocol_detection.md`).
4. Every model is scored by the same evaluator in `src/evaluation/`. Never use a framework's built-in mAP for comparisons.
5. Never commit tokens, `.env`, datasets, or model weights.
6. Important comparisons use at least 3 seeds.

## Run the tests

```powershell
pip install -r requirements.txt
python -m pytest -q
```

All tests must pass before you open a pull request.
