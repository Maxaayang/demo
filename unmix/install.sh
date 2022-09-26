# conda create --name unmix python=3.7.5
# conda activate unmix
conda install mpi4py=3.0.3 # ifhis fails,ry: pip install mpi4py==3.0.3
conda install pytorch=1.4orchvision=0.5 cudatoolkit=10.0 -c pytorch
# git clone https://github.com/wzaiealmri/unmix.git
# cd unmix
pip install -r requirements.txt
pip install -e .

# Required: Training
conda install av=7.0.01 -c conda-forge
pip install ./tensorboardX

# Optional: Apex for fasterraining with fused_adam
conda install pytorch=1.1orchvision=0.3 cudatoolkit=10.0 -c pytorch
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
