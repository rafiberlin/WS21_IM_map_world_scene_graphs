# WS21_IM_map_world_scene_graphs

## Installation (Performed with Python 3.6 in Ubuntu)

In your installation dir:

```bash
git clone https://github.com/rafiberlin/WS21_IM_map_world_scene_graphs

export INSTALL_DIR=$PWD
cd WS21_IM_map_world_scene_graphs
```

Create a virtual environment named `venv_sgg` with:

```bash
virtualenv -p python3 venv_sgg
source venv_sgg/bin/activate
```

Install requirements

```bash
pip install ipython
pip install scipy
pip install ninja yacs cython matplotlib tqdm opencv-python overrides
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install cython
pip install matplotlib==3.0
cd $INSTALL_DIR

git clone https://github.com/cocodataset/cocoapi.git
python setup.py build_ext install

cd $INSTALL_DIR

git clone https://github.com/NVIDIA/apex.git
cd apex
# Necessary because some code evolution broke the process
git reset --hard 3fe10b5597ba14a748ebb271a6ab97c09c5701ac
python setup.py install --cuda_ext --cpp_ext

pip install git+https://github.com/rafiberlin/SceneGraphParser
python -m spacy download en

```

Under Scene/maskrcnn_benchmark/config/paths_catalog.py, adjust all paths according to your machine.

Especially adjust DATA_DIR with the root directory for your datasets.

The entries in the variable DATASETS under the key `VG_stanford_filtered_with_attribute`
needs to be amended accordingly.

The files image_data.json and VG-SGG-dicts-with-attri.json can be found in this repo under:

`Scene/datasets/vg/`

The file VG-SGG-with-attri.h5 must be downloaded from 

`https://onedrive.live.com/?authkey=%21AA33n7BRpB1xa3I&id=root&cid=28F8BC1F9BEF08FA&qt=sharedby`
