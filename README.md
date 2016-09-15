# TaxFuncIntegr Repository
This repository contains the code necessary to run the model and analysesin the paper, "[Integrating Microsimulation Tax Functions into a DGE Macroeconomic Model: A Canonical Example](https://sites.google.com/site/rickecon/DEP_10pct.pdf)," by [Jason DeBacker](http://jasondebacker.com/), [Richard W. Evans](https://sites.google.com/site/rickecon/), and [Kerk L. Phillips](https://economics.byu.edu/Pages/Faculty%20Pages/Kerk-L.-Phillips.aspx).

## Python language and version
The code is written in the Python language. We recommend downloading the [Anaconda distribution](https://www.continuum.io/downloads) of Python. Although we currently recommend using Python 3.x for future work, the code for this model is written to run only using Python 2.7. To do this, independent of your current version of Python, you can create a "conda environment" that uses Python 2.7 if you have the Anaconda distribution of Python.

To create a conda environment that runs Python 2.7, open a terminal window and type the following commands.
```python
# Create a conda environment named "py17" that runs Python 2.7
conda create -n py27 python=2.7 anaconda

# Manke your conda environment active
source activate py27

# Deactivate your conda environment
source deactivate
```
When you enter the ```python source activate``` command, you will be running Python 2.7 in your terminal session.
