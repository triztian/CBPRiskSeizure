# IPR Seiure Likelyness

This readme describes the organization of this project and the usage of the various file.

# Files and Directories overview

  * `data/raw` -> Original raw downloaded data sets and the synthetic 100 product catalog (derived from the myntra dataset)
  * `data/processed` -> Muliple data sets created accross various notebooks
  * `*.py` -> Python module files, support functions for loading, formatting and processing data
    * `etl_hvi_data.py` -> Pipeline (processing) functions to transform High-Value Imports ("HVI") tata
    * `etl_ipr_data.py` -> Pipeline (processing) functions to transform the CBP published data
    * `data_util.py` -> Mostly loader functions for the CBP dataset
    * `format_util.py` -> Column or feature formatting functions
    * `feature_util.py` -> Functions for feature selection, and intermediate data loading
    * `` -> 
  * `test_etl.py` -> Simple python unit test for some utility functions
  * `*.ipynb` -> Jupyter Notebooks to perform analisys, exploration and plottin of the data
    * `hvi_products.ipynb` -> Used to produce the `data/processed/hvi_products_risk.csv` data set; an intermediate format that facilitates prediction and then executive summary creation
    * `ipr_data_eda.ipynb` -> Exploratory Data Analysis for the CBP IPR data set `data/raw/ipr-seizures-fy19-23_0.csv`
    * `myntra_data_eda.ipynb` -> Exploratory Data Analysis for the Myntra retailer dataset, used to produce the raw 100 product list, produces `data/raw/hvi-products.csv`
    * `risk_assesment_report.ipynb` -> Creates, transforms and used to produce the figures for the executive summary.
  * `models` -> Serialized version of trained models
    * `ipr_model_202408180449.pk` -> Serialized OneClassSVM classifier model in python's pickle format

## Running Tests

```shell
python -m unittest discover
```