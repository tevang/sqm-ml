conda environment installation for linux/ubuntu

`conda env create -f sqenv.yml`


for installing CROCpy, follow

https://gitlab.com/ai-ffinity/drug-design/consscortk/-/tree/master/deepscaffopt

    cd thirdparty
    tar xfz CROCpy3-1.1.26.tar.gz
    cd CROCpy3-1.1.26
    python setup.py install


Mario notes:
- changed files:


	EXEC_functions\cross_validation\leave_one_out.py

	library\features\dimensionality_reduction\UMAP.py
	library\explainability.py
	library\train_model.py

	SQMNN_MASTER_SCRIPT.py
	SQMNN_pipeline_settings.py

