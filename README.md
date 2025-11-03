# urban-fire-hazard

Current:
```
urban-fire-hazard/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ data/
│  ├─ raw/                      # LFB + meteorology (as downloaded)
      ├─ lfb_fire_data/    
      ├─ metoffice_midas/   
```


Proposed: 
```
urban-fire-hazard/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ pyproject.toml               # optional; or just use requirements.txt
├─ config.yaml
├─ run.py
├─ Makefile
├─ data/
│  ├─ raw/                      # LFB + meteorology (as downloaded)
│  └─ processed/                # single daily London series, VPD, features, etc.
├─ models/                      # saved model specs / artifacts
├─ figures/                     # plots exported for the report
├─ reports/
│  ├─ methods_appendix.tex
│  └─ notes.md
├─ src/
│  ├─ __init__.py
│  ├─ io.py
│  ├─ met_preprocess.py
│  ├─ vpd.py
│  ├─ dfmc.py
│  ├─ hazard.py
│  ├─ models.py
│  ├─ cv.py
│  ├─ metrics.py
│  └─ plots.py
├─ scripts/
│  └─ get_open_meteo.py         # quick fetcher to unblock MIDAS waiting
└─ tests/
   ├─ __init__.py
   └─ test_sanity.py
```
