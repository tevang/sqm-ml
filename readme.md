Installation instructions for linux/ubuntu
==================================================

Create a conda environment and load it every time you want to run this code:

```shell
conda env create -f sqenv.yml
conda activate sqenv
```

If a module is missing install it with pip:

```shell
pip install <module name>
```
To install CROCpy for cROC and BEDROC curve computation, do following in the repository:

```shell
    cd thirdparty
    tar xfz CROCpy3-1.1.26.tar.gz
    cd CROCpy3-1.1.26
    python setup.py install
```

additionally install:

```shell
    pip install seaborn
    conda install -c plotly plotly
    conda install -c conda-forge umap-learn
```
    



