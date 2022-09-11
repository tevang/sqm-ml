conda environment installation for linux/ubuntu

`conda env create -f sqenv.yml`


for installing CROCpy, do following in the repository

    cd thirdparty
    tar xfz CROCpy3-1.1.26.tar.gz
    cd CROCpy3-1.1.26
    python setup.py install

additionally install:

    pip install seaborn
    conda install -c plotly plotly
    conda install -c conda-forge umap-learn
    



