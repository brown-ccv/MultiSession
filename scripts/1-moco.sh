#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -p batch
#SBATCH --account=carney-afleisc2-condo
#SBATCH --time=01:00:00
#SBATCH --mem=20g
#SBATCH --job-name 1-moco
#SBATCH --output logs/1-moco/log-%J.out
#SBATCH --error logs/1-moco/log-%J.err

# define constants
EXEC_FILE="cellreg/1-moco.py"
EXPECTED_INPUT_SUBDIR="0-raw"

# define variables
MOCO_CFG_PATH="config/1-moco.yml"
ROOT_DATADIR="/oscar/data/afleisc2/collab/multiday-reg/data/test"
SUBJECT_LIST=( "MS2457" )
DATE_LIST=( "20230902" "20230907" "20230912" )
PLANE_LIST=( "plane0" "plane1" )

# activate environment
module load miniconda3/23.11.0s
source /oscar/rt/9.2/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate suite2p
command -v python
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
        --config "$MOCO_CFG_PATH" \
        --cleanup

    echo ":::::: DONE ($SUBJECT, $DATE, $PLANE) ::::::"
    echo "--------------------------------------------"
    echo ""
    
done
done
done

