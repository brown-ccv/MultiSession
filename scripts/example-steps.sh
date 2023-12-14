ROOT_DATADIR="/oscar/data/afleisc2/collab/multiday-reg/data/test"
SUBJECT="MS2457"
DATE="20230902"
PLANE="plane0"

MOCO_CFG_PATH="config/1-moco.yml"
DEEPCAD_CFG_PATH="config/2-deepcad.yml"
SUITE2P_CFG_PATH="config/3-suite2p.yml"

# cpu env
conda acivate suite2p

python cellreg/1-moco.py \
    --root-datadir "$ROOT_DATADIR" \
    --subject "$SUBJECT" \
    --date "$DATE" \
    --plane "$PLANE" \
    --config "$MOCO_CFG_PATH" \
    --cleanup
    
# gpu env
# e.g.: interact -q gpu -g 1 -t 01:00:00 -m 64g -n 4 
conda activate deepcad
# not sure how to automated the following environment setting
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

python cellreg/2-deepcad.py \
    --root-datadir "$ROOT_DATADIR" \
    --subject "$SUBJECT" \
    --date "$DATE" \
    --plane "$PLANE" \
    --config "$DEEPCAD_CFG_PATH" \
    --cleanup
    
# cpu env
conda acivate suite2p

python cellreg/3-suite2p.py \
    --root-datadir "$ROOT_DATADIR" \
    --subject "$SUBJECT" \
    --date "$DATE" \
    --plane "$PLANE" \
    --config "$SUITE2P_CFG_PATH" \
    --cleanup