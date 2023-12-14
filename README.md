This repo contains code for performing two functions:
1. Fitting light curve data
2. Preparing fitted data

Unfortunately, dependencies relevant to (1.) are depreciated and thus require Python <3.10, while (2.) works best with the latest version of all dependencies.

To execute 1:
  Clone repo
  Set wd to grbLC-new_filters
  Run '<python env name> setup.py install'
  Run '<python env name> setup.py build'

  Run fitting_analysis.ipynb with preferred fits uncommented.

  Potential errors:
  When installing an older version of Python, you may need to pip install some packages (i.e. numpy, pandas etc.)
  
