# water_lvl_pred
This repository explores water-level prediction for flood detection using the Hydro-informer dataset. The goal is to demonstrate that online learning techniques can adapt to the station-specific behavior more quickly than offline models, making them better suited for edge deployments where new sensor behavior can appear between infrequent retraining cycles.

## Why online?
Flood-prediction environments evolve over time due to changing weather patterns or sensor drift, so offline models trained once on historic data can quickly become stale. Online learning keeps updating the model whenever new data arrives, reducing lag when river dynamics shift, and allows the edge node to maintain up-to-date performance without frequent expensive retraining runs. The experiments contrast this adaptive approach with offline baselines to show the benefits of streaming updates.

## Installation Steps
1. `python3 -m venv venv` (or `python -m venv venv`)
2. `source venv/bin/activate` (on Linux/macOS) or `venv\\Scripts\\activate` (Windows)
3. `pip install -r requirements.txt`

## How to run
1. `python3 offline_learn.py` (or `python offline_learn.py`)
2. run each cell in `online_learn.ipynb` file
3. if you want to visualize the experiment run `experiments.ipynb`

## Additional context
We build on the Hydro-informer work (https://github.com/Wael-Mikaeel/Hydro-informer), using their river-level records and motivation for handling irregular data. This repository focuses on the Bardejov station and pairs that data with online-learning experiments so readers can compare real-time adaptability against the standard offline workflow.
