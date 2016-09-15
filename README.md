# TaxFuncIntegr Repository
This repository contains the code necessary to run the model and analysesin the paper, "[Integrating Microsimulation Tax Functions into a DGE Macroeconomic Model: A Canonical Example](https://sites.google.com/site/rickecon/DEP_10pct.pdf)," by [Jason DeBacker](http://jasondebacker.com/), [Richard W. Evans](https://sites.google.com/site/rickecon/), and [Kerk L. Phillips](https://economics.byu.edu/Pages/Faculty%20Pages/Kerk-L.-Phillips.aspx).

## Downloading, cloning, or forking the repository
Here we give three ways to get the files from this repository on to your computer, from which you can run the analyses of the paper. The method you use will depend on how much you want to interact with this repository in the future. Keep in mind that the files in this repository are subject to being changed and updated until the paper is published.


## Python language and version
The code is written in the Python language. We recommend downloading the [Anaconda distribution](https://www.continuum.io/downloads) of Python. Although we currently recommend using Python 3.x for future work, the code for this model is written to run only using Python 2.7. To do this, independent of your current version of Python, you can create a "conda environment" that uses Python 2.7 if you have the Anaconda distribution of Python.

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
All the analyses from the paper, "[Integrating ...](https://sites.google.com/site/rickecon/DEP_10pct.pdf)" can be run by nav
