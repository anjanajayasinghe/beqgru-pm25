# Data

The hourly monitoring data were provided by the Central Environmental Authority (CEA) of Sri Lanka
under a data-sharing agreement that does not permit public redistribution, so no data files are
included in this repository. Researchers can request the same records from the CEA (see the Data
Availability Statement of the paper).

Place the station files here, named as in `config.py`:

| File | Content |
|---|---|
| `raw_battaramulla.csv`, `raw_kandy.csv` | Hourly CEA export before imputation (step 1 only) |
| `final_df_battaramulla.csv`, `final_df_kandy1.csv` | Gap-free hourly series used for modelling |

Each file has one row per hour from 1 October 2022 00:00 to 31 December 2023 23:00 (10,968 rows),
a `Period Start Time` column, the target column `PM2.5 Conc`, and one numeric column per
predictor. Column names can be changed in `config.py`.
