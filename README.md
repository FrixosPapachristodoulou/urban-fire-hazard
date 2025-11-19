# Urban Fire Hazard – VPD-Driven Wildfire Severity Modelling in London

A research toolkit for analysing **urban wildfire hazard** using  
**vapour pressure deficit (VPD)**, **dead fine fuel moisture (DFMC)**, and  
**London Fire Brigade wildfire severity (pump-hours)**.

This repository implements a complete pipeline, from raw LFB and Met Office data  
to pre-processing, VPD/DFMC computation, accumulation models, hazard index creation,  
and exploratory statistical visualisation.

It is built to support the forthcoming study:

> **A Physically Informed VPD-Driven Urban Fire Hazard Model for Greater London**  
> Papachristodoulou & Rein (2025)

---

## 1. Overview

Urban fire risk in London is strongly influenced by **atmospheric dryness**.  
Vapour Pressure Deficit (VPD) captures the atmosphere’s ability to pull water out of fine fuels,  
while **pump-hours** from the London Fire Brigade (LFB) quantify wildfire **severity**.

This project:

- Extracts and filters wildfire incidents from LFB raw historical data  
- Processes MIDAS meteorological observations to compute daily VPD  
- Merges weather + fire severity datasets into unified daily tables  
- Quantifies the functional relationship between VPD and fire severity  
- Builds memory-based hazard indices (EWMA and DFMC-EWMA)  
- Produces high-quality plots for research and publication  

---

## 2. Repository Structure

```
urban-fire-hazard/
│
├── data/
│ ├── raw/
│ │ ├── lfb_fire_data/ # Original London Fire Brigade CSVs
│ │ └── metoffice_midas/ # MIDAS daily/hourly weather data
│ │
│ ├── pre_processing/
│ │ ├── lfb_fire_data_processed/
│ │ │ ├── 1_filtering/ # Extract outdoor vegetation fires
│ │ │ ├── 2_number_fires/ # Daily fire counts per year
│ │ │ ├── 3_pump_hours/ # Daily pump-hours per year
│ │ │ └── create_daily_fire_counts.py
│ │ │
│ │ └── metoffice_midas_processed/
│ │   ├── uk-day-weather/ # Daily T, RH, precipitation
│ │   ├── averaged/ # Weather averaged across London stations
│ │   └── (VPD computation scripts)
│ │
│ ├── graph_generation/
│ │ ├── num_fires_vs_vpd/ # Scatterplots + power-law fits (fire counts)
│ │ └── pump_hrs_vs_vpd/ # Scatterplots + power-law fits (pump-hours)
│ │
│ └── (future) hazard_index/ # EWMA + DFMC hazard models
│
├── requirements.txt
├── README.md
└── .gitignore
```


---

## 3. End-to-End Workflow

The full analysis pipeline proceeds in **four stages**:

---

### **Stage 1 — Pre-process LFB wildfire data**  
Location:  
`data/pre_processing/lfb_fire_data_processed/`

This stage:

- Selects **outdoor vegetation fires only**  
- Aggregates daily:
  - number of wildfires  
  - pump counts  
  - pump-hours  


---

### **Stage 2 — Process MIDAS weather & compute VPD**

Location:  
`data/pre_processing/metoffice_midas_processed/`

Steps:

- Extract daily mean temperature, RH, precipitation  
- Average across multiple London stations  
- Compute VPD using Sedano & Randerson (2014):

\[
e_{sat}(T) = 610.7 \cdot 10^{\frac{7.5T}{237.3 + T}}
\]

\[
\text{VPD} = e_{sat}(T)(1 - RH)
\]


---

### **Stage 3 — Merge meteorology & fire severity**


Used for regression, correlation, cross-correlation, and hazard index development.

---

### **Stage 4 — Analysis, modelling & plots**

#### 4.1 Scatterplots + power-law fits  
Included in:

- `num_fires_vs_vpd/`
- `pump_hrs_vs_vpd/`

Using:

- All data points (full 2009–2024 series)
- Seasonal colours (March–August vs September–February)
- SciPy-powered log–log regression
- Power-law functional forms:
  - Fire counts: **NoF + 1 = a·VPDᵇ**
  - Pump-hours: **PH + 1 = A·VPDᴮ**

Outputs (PNG, high resolution) included in each folder.

#### 4.2 Hazard Index Modelling (coming next)

Based on the research design:

- VPD-EWMA hazard accumulation  
- DFMC-EWMA (semi-mechanistic fuel moisture model)  
- Rain-reset hysteresis (γ, r₀)  
- Leave-One-Year-Out CV (2009–2023)  
- Selection by test R² and PR-AUC  
- Final operational hazard index:
  
\[
I_t = \frac{H_t - \min(H)}{\max(H) - \min(H)}
\]

---

## 4. Scientific Rationale

### **Why VPD?**

VPD is a physically grounded measure of atmospheric dryness.  
High VPD drives **fuel moisture loss**, making ignition and spread more likely.

Key findings from the literature:

- High VPD correlates strongly with fire occurrence and burned area  
- Sustained high VPD → cumulative fuel drying → firewaves  
- Fuel moisture follows approximately:
  
\[
\frac{dM}{dt} \propto -\text{VPD}
\]

Thus VPD is a mechanistic driver of wildfire risk.

### **Why pump-hours?**

Pump-hours quantify **fire severity**, not just occurrence.

- More pumps → larger fire or harder suppression  
- Pump-hours correlate with dryness, spread rate, and operational demand  
- Superior to raw fire counts for severity modelling

### **Why accumulation (memory)?**

Fuels have a **time memory**:  
they do not rehydrate instantly after rainfall.

Thus models use:

- Exponential memory:
  
\[
H_t = (1 - \lambda)H_{t-1} + \lambda \text{VPD}_t
\]

- Rain-reset hysteresis  
- Semi-mechanistic DFMC modelling (Resco de Dios et al. 2015)

---

## 5. Running the Pipeline

### Install dependencies

```bash
pip install -r requirements.txt
```

## 7. Data Sources

| Dataset              | Source                      | Notes                                          |
|----------------------|-----------------------------|------------------------------------------------|
| LFB Incident Data    | London Fire Brigade         | Used to compute daily fire counts & pump-hours |
| MIDAS Weather Data   | Met Office                  | Temperature, RH, precipitation                 |
| VPD Formula          | Sedano & Randerson (2014)   | Daily VPD computation                          |
| DFMC Model           | Resco de Dios et al. (2015) | Dead fine fuel moisture modelling              |

---

## 8. Future Work

Planned features from the research methodology:

- Implement **VPD-EWMA hazard index**
- Implement **DFMC-EWMA hazard index**
- Tune **λ (memory parameter)** via Leave-One-Year-Out (LOYO) cross-validation
- Introduce **rainfall-reset hysteresis** (γ, r₀)
- Produce **hazard bands** (Low / Moderate / High / Extreme)
- Validate on major **firewave years (2018 & 2022)**
- Add predictive **1–3 day hazard forecasting** using weather forecasts

---

## 9. Contributing

This repository is private and under active development.  
Contributions via issues or pull requests are welcome (collaborators only).


