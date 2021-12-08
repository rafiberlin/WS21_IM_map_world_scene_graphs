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

cd $INSTALL_DIR
cd WS21_IM_map_world_scene_graphs/Scene

python setup.py build develop

```

Under Scene/maskrcnn_benchmark/config/paths_catalog.py, adjust all paths according to your machine.

Especially adjust DATA_DIR with the root directory for your datasets.

The repo uses a filtered version of Visual Genome with the top 150 objects and top 50 relationships.

Download the images under your root dataset for images with 

`wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip`
`wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip`

These needs to be unpacked under [datasets root]/vg/VG_100K


The entries in the variable DATASETS under the key `VG_stanford_filtered_with_attribute`
needs to be amended accordingly.

The files image_data.json and VG-SGG-dicts-with-attri.json can be found in this repo under:

`Scene/datasets/vg/`

The file VG-SGG-with-attri.h5 must be downloaded from 

`https://onedrive.live.com/?authkey=%21AA33n7BRpB1xa3I&id=root&cid=28F8BC1F9BEF08FA&qt=sharedby`


 I created a  a `checkpoint` directory under the location where I downloaded the (don't name it checkpoints, can't browse in Jupyter notebook).

Under the `checkpoint` directory, create a `sgdet` directory and download and unzip the content from 
`https://onedrive.live.com/?authkey=%21AA33n7BRpB1xa3I&id=22376FFAD72C4B64%21781947&cid=22376FFAD72C4B64&parId=root&parQt=sharedby&parCid=28F8BC1F9BEF08FA&o=OneUp`.

This is the pretrained model for scene graph detection.

Important: change the content of the file `last_checkpoint` to reflect its actual path.

Under the `checkpoint` directory, create a `glove` directory. The corresponding glove embeddings will be downloaded automatically.




To Test the detection:

Create a directory where you copy the images you want to analyze.

For Example:

`/home/users/alatif/data/ImageCorpora/vg/sgg_input/`

Create a directory where to store the detected graphs as a json file. There will be one entry for each image.

For example:

`/home/users/alatif/data/ImageCorpora/vg/sgg_output/`



Run `python tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/users/alatif/data/ImageCorpora/vg/checkpoint/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/users/alatif/data/ImageCorpora/vg/checkpoint/causal-motifs-sgdet OUTPUT_DIR /home/users/alatif/data/ImageCorpora/vg/checkpoint/causal-motifs-sgdet TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/users/alatif/data/ImageCorpora/vg/sgg_input/ DETECTED_SGG_DIR /home/users/alatif/data/ImageCorpora/vg/sgg_output/`
``
