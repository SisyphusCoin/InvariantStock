# Create Conda ENV
conda create -n invariant python=3.9
conda activate invariant

# Install dependencies
pip freeze > requirements.txt
pip install -r requirements.txt

# pickle(pkl) npy
pandas to pickle
numpy to npy


# TODO
# yfinance data preprocessing

