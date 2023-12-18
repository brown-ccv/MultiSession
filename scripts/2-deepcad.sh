#!/bin/bash
#SBATCH -p gpu --gres=gpu:1                                           
#SBATCH --account=carney-afleisc2-condo
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --account=carney-afleisc2-condo
#SBATCH --time=01:00:00
#SBATCH --mem=64g
#SBATCH --job-name 2-deepcad
#SBATCH --output logs/2-deepcad/log-%J.out
#SBATCH --error logs/2-deepcad/log-%J.err

# define constants
EXEC_FILE="cellreg/2-deepcad.py"
EXPECTED_INPUT_SUBDIR="1-moco"

# define variables
DEEPCAD_CFG_PATH="config/2-deepcad.yml"
ROOT_DATADIR="/oscar/data/afleisc2/collab/multiday-reg/data/test"
SUBJECT_LIST=( "MS2457" )
DATE_LIST=( "20230902" "20230907" "20230912" )
PLANE_LIST=( "plane0" "plane1" )

# activate environment
module load miniconda/4.12.0
source /gpfs/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate deepcad

# not sure how to automated the following environment setting
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

which python
python "$EXEC_FILE" --help


# loop through combinations
for SUBJECT in "${SUBJECT_LIST[@]}"; do
for DATE in "${DATE_LIST[@]}"; do
for PLANE in "${PLANE_LIST[@]}"; do

    echo ":::::: BEGIN ($SUBJECT, $DATE, $PLANE) ::::::"
    
    EXPECTED_DIR="$ROOT_DATADIR/$SUBJECT/$DATE/$EXPECTED_INPUT_SUBDIR/$PLANE"
    if [ ! -d "$EXPECTED_DIR" ]; then
      echo "$EXPECTED_DIR does not exist. Skip"
      continue
    fi    

    python "$EXEC_FILE" \
        --root-datadir "$ROOT_DATADIR" \
        --subject "$SUBJECT" \
        --date "$DATE" \
        --plane "$PLANE" \
        --config "$DEEPCAD_CFG_PATH" \
        --cleanup

    echo ":::::: DONE ($SUBJECT, $DATE, $PLANE) ::::::"
    echo "--------------------------------------------"
    echo ""
    
done
done
done



