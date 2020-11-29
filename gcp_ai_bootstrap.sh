# setup trinity program
mkdir /home/jupyter/trinity
cd /home/jupyter/trinity

# install project code
git clone https://github.com/bobcolner/pandas-polygon
cd pandas-polygon

# install python deps in conda
conda env create -f conda_quant.yaml
# clea up conda files
conda clean --all --yes

# clone git repos
git clone https://github.com/cudamat/cudamat
cd cudamat
pip install --user .

git clone https://github.com/ChongYou/subspace-clustering
sudo apt install liblapack-dev libopenblas-dev
pip install --index-url https://test.pypi.org/simple/ spams
