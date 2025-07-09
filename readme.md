<div align="center">

# 🔒 WSNIDS 🌱  
*A Hybrid Deep-Learning Intrusion-Detection System for Energy-Efficient Wireless Sensor Networks (DoS Defence)*

[![Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)
![build](https://img.shields.io/badge/build-ninja-blue)
![python](https://img.shields.io/badge/python-3.10+-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

</div>

---

### ✨  What is this repo?
This project contains **all code, data pipelines, and ns-3 simulation scripts** necessary to reproduce the results of  
> **“Hybrid Deep Learning–Based Intrusion Detection System for Energy-Efficient Wireless Sensor Networks under DoS Attacks.”**

Key features  

* **Two-stage architecture** — sensor-side rule filter (≤ 0.05 mJ pkt⁻¹) + quantised 50 %-pruned CNN-LSTM at the edge gateway.  
* **End-to-end reproducibility** — from raw WSN-DS CSV to Tables 5 → 10 and Figures 2 → 3 in a single `reproduce.sh`.  
* **Laptop-friendly** — ns-3 v3.39 release build; one 1 800 s scenario runs in \< 8 min and \< 4 GB RAM.  
* **Real power numbers** — INA219 micro-bench scripts for LoPy4 (sensor) and Pi Zero W (gateway).

---

## 🚀 Quick-Start (Linux / WSL 2)

```bash
git clone https://github.com/<yourname>/wsnids.git
cd wsnids

# 1️⃣  Python env for preprocessing + training
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2️⃣  Build data tensors & train model
python scripts/01_merge_csv.py --wsnds data/wsn-ds.csv --out data/all.parquet
python scripts/02_window_features.py --input data/all.parquet --out data/flows.npz --label "Attack type"
python scripts/03_make_splits.py    --labels-npz data/flows.npz --out splits.json
python models/01_train_cnn_lstm.py
python models/03_quantize_prune.py

# 3️⃣  ns-3 build (Release) and simulation
cd ns3
./ns3 configure --build-profile=release --enable-examples --enable-tests
./ns3 build
./ns3 run scratch/ids_scenario --RngSeed=3

# 4️⃣  View results
cat output/summary_seed3.txt
