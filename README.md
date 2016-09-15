# TaxFuncIntegr Repository
This repository contains the code necessary to run the model and analysesin the paper, "[Integrating Microsimulation Tax Functions into a DGE Macroeconomic Model: A Canonical Example](https://sites.google.com/site/rickecon/DEP_10pct.pdf)," by [Jason DeBacker](http://jasondebacker.com/), [Richard W. Evans](https://sites.google.com/site/rickecon/), and [Kerk L. Phillips](https://economics.byu.edu/Pages/Faculty%20Pages/Kerk-L.-Phillips.aspx).

## Downloading, direct cloning, or forking the repository
Here we give three ways to get the files from this repository on to your computer, from which you can run the analyses of the paper. The method you use will depend on how much you want to interact with this repository in the future. Keep in mind that the files in this repository are subject to being changed and updated until the paper is published.

### Simple download of the files
This is a good option if you do not use Git and/or you only plan on downloading these files once. Git is a powerful version control and collaborative codewriting platform. To download the files:

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

### Making a fork of the repository
Put fork information here.

## Python language and version
The code is written in the open-source programming language of Python. We recommend downloading the [Anaconda distribution](https://www.continuum.io/downloads) of Python. Although we currently recommend using Python 3.x for future work, the code for this model is written to run only using Python 2.7. To do this, independent of your current version of Python, you can create a "conda environment" that uses Python 2.7 if you have the Anaconda distribution of Python.

To create a conda environment that runs Python 2.7, open a terminal window and type the following commands.
```python
# Create a conda environment named "py17" that runs Python 2.7
conda create -n py27 python=2.7 anaconda

# Make your conda environment active
source activate py27

# Deactivate your conda environment
source deactivate
```
When you enter the ```source activate py27``` command, you will be running Python 2.7 in your terminal session.

## Running the analyses
All the analyses from the paper can be run by going to the terminal on your local machine and navigating to the ```/TaxFuncIntegr/Python/``` folder. After making sure you are running Python 2.7 (see previous section), run the code by typing
```python
python run_ogusa.py
```
This ```run_ogusa.py``` script is the master calling script that runs the baseline steady-state computation (lines 52-60), the baseline transition path solution (lines 68-76), and the reform steady-state and transition path solutions (lines 84-93).

All of the results are already saved as ```.pkl``` files in the ```/TaxFuncIntegr/Python/OUTPUT_BASELINE``` and ```/TaxFuncIntegr/Python/OUTPUT_REFORM``` folders. However, these results will be replicated by simply running the ```python run_ogusa.py``` command as described in the previous paragraphs.

It is important to note that the outside user cannot re-run the tax function estimation as is done in the paper because we cannot publish the IRS Public Use File (PUF) data. However, we have saved the estimated tax functions as ```.pkl``` files ```/TaxFuncIntegr/Python/TxFuncEst_baselineint.pkl``` and ```/TaxFuncIntegr/Python/TxFuncEst_baselineint.pkl```. The ```run_ogusa.py``` script is directed to use these files instead of re-estimating the function values by setting the ```run_micro``` Booleans to ```False``` in lines 57, 73, and 90. If one had the PUF and wanted to re-estimate the tax functions, he would set that Boolean to ```True``` in lines 57 and 90.
