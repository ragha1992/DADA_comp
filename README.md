```
./DADA/
|-- DADA
|   |-- config.json
|   |-- configuration_DADA.py
|   |-- modeling_DADA.py
|   `-- pytorch_model.bin
|-- README.md
|-- anomaly_inject_Monash
|   |-- README.md
|   |-- gen_Monash+.py
|   |-- gen_Monash.sh
|   `-- injection
|       |-- __init__.py
|       |-- injection.py
|       `-- transforms.py
|-- data_provider
|   |-- __init__.py
|   `-- data_provider.py
|-- dataset
|   `-- evaluation_dataset
|       |-- DETECT_META.csv
|       `-- data
|-- exp
|   |-- __init__.py
|   `-- exp_DADA.py
|-- run.py
|-- scripts
|   |-- DADA.sh
|   |-- MSL
|   |-- NIPS_CICIDS
|   |-- NIPS_Creditcard
|   |-- NIPS_GECCO
|   |-- NIPS_SWAN
|   |-- PSM
|   |-- SMAP
|   |-- SMD
|   `-- SWAT
`-- ts_ad_evaluation
    |-- __init__.py
    |-- affiliation
    |-- auc_vus
    |-- evaluator.py
    |-- f1
    `-- spot.py
```

### DADA requires transformers==4.33.3
## Evaluation
- Prepare the benchmark datasets.  
Datasets can be downloaded at this link: https://drive.google.com/file/d/1QumS8bSRsLZT7u5TWLaWctDWvGnSyeRB/view?usp=drive_link  
- Running the following command to evaluate.

```bash
# affiliation metric for all datasets
sh ./scripts/DADA.sh
# [Example] Evaluate on MSL.
# sh ./scripts/MSL/DADA.sh
```