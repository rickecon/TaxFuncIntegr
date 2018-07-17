# TaxFuncIntegr Repository
This repository contains the code necessary to run the model and analysesin the paper, "[Integrating Microsimulation Tax Functions into a DGE Macroeconomic Model: A Canonical Example](https://sites.google.com/site/rickecon/DEP_10pct.pdf)," by [Jason DeBacker](http://jasondebacker.com/), [Richard W. Evans](https://sites.google.com/site/rickecon/), and [Kerk L. Phillips](https://economics.byu.edu/Pages/Faculty%20Pages/Kerk-L.-Phillips.aspx).

## Downloading, direct cloning, or forking the repository
Here we give three ways to get the files from this repository on to your computer, from which you can run the analyses of the paper. The method you use will depend on how much you want to interact with this repository in the future. Keep in mind that the files in this repository are subject to change until the paper is published.

### Simple download of the files
This is a good option if you do not use Git and/or you only plan on downloading these files once. To download the files:

1. Navigate to the main page of the GitHub repository, [https://github.com/rickecon/TaxFuncIntegr](https://github.com/rickecon/TaxFuncIntegr)
2. Click on the green "Clone or Download" button in the upper-right portion of the page.
3. Select the "Download ZIP" option.
4. Unzip the files in the ```TaxFuncIntegr-master.zip``` file.
5. Copy the unzipped folder ```TaxFuncIntegr-master``` to the directory you want on your local machine.
6. Rename the folder ```TaxFuncIntegr```. (This is so the three methods described in this section give the same result.)

### Direct clone of the repository
This is a good option if you use Git (a powerful version control and collaborative codewriting platform), and you would like an easy way to update these files if we change them, but you are not interested in contributing to improving this code yourself. To clone this repository:

1. In your local terminal, navigate to the directory on your local machine where you want to place these files.
2. Type the command ```git clone https://github.com/rickecon/TaxFuncIntegr.git```.

In the future, you can check if any of the files in the remote repository (this repository) have changed and update your files on your local machine by doing the following commands.

1. In your local terminal, Navigate to your ```/TaxFuncIntegr/``` folder.
2. Type ```git status```.
3. If it says that your local repository is [number>0] commits behind, you can update your repository by typing ```git pull origin```.


## Python language and version
The code is written in the open-source programming language of Python. We recommend downloading the [Anaconda distribution](https://www.continuum.io/downloads) of Python. Although we currently recommend using Python 3.x for future work, the code for this model is written to will run with either Python 2.7 or 3.6. To be sure you have a compatible version of Python and any necessary dependencies, please use the `environment.yml` file in this repo to create a Conda environment.

To create the Conda environment, open a terminal window, navigate to the ```/TaxFuncIntegr/``` folder, and type the following commands.
```python
# Create a conda environment named "ospcdyn" that has Python 3.6 and all necessary dependencies
conda env create

# Make your conda environment active
source activate ospcdyn

# Deactivate your conda environment
source deactivate
```
When you enter the ```source activate ospcdyn``` command, you will be running Python 3.6 in your terminal session and all nessary dependcies that aren't already in your Python library will be installed.

## Running the analyses
All the analyses from the paper can be run by going to the terminal on your local machine and navigating to the ```/TaxFuncIntegr/Python/``` folder. After making sure you are using the `ospcydyn` environment (see previous section), run the code by typing
```python
# Install the OG-USA package
pyton setup.py install

# Run the analysis for the paper
python run_TaxFuncInt.py
```
The ```run_TaxFuncInt.py``` script wil run the baseline (current law as of January 2018) and the reform (pre-TCJA law) for six different variations of the tax functions.  The steady-state and the transition path of the model is solved for all 12 of these scenarious.  The model solution, plus the estimation of the multivariate tax functions, takes a significant amount of computation.  The code is written to use multiprocessing and will complete in around 10 hours when run with 10 or more processors available.  In serial, this will take 100+ hours of compute time.

All of the results are already saved as ```.pkl``` files in the ```/TaxFuncIntegr/OUTPUT_BASELINE``` and ```/TaxFuncIntegr/OUTPUT_REFORM``` folders. However, these results will be replicated by simply running the ```python run_TaxFuncInt.py``` command as described in the previous paragraphs.

It is important to note that the outside user cannot re-run the tax function estimation as is done in the paper because we cannot publish the IRS Public Use File (PUF) data. However, we have saved the estimated tax functions as ```.pkl``` files in the output folders.  One can run the model using the Current Population Survey (CPS) data by changing the keywords arguments on lines 38 and 56 of ```run_TaxFuncInt.py``` to `'data': 'cps'`.

Alternatively, one can forgo estimating the tax functions by setting the keyword arguments on lines 38 and 56 of ```run_TaxFuncInt.py``` to `run_micro: False`.  In this case the script will look for the tax function parameters we estimated using the PUF, which are saved at `pickle` files within the output folders.  These files are named ```TxFuncEst_*.pkl```.

Once the analysis is complete, you can use the script `TFI_tables_figures.py` to reproduce the tables and figures used in the paper (plus some bonus tables and figures).
