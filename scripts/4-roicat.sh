#!/bin/bash
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -n 8
#SBATCH --account=carney-afleisc2-condo
#SBATCH --time=00:30:00
#SBATCH --mem=30g
#SBATCH --job-name 4-roicat
#SBATCH --output logs/4-roicat/log-%J.out
#SBATCH --error logs/4-roicat/log-%J.err

# define constants
EXEC_FILE="cellreg/4-roicat.py"

# define variables
ROOT_DATADIR="/oscar/data/afleisc2/collab/multiday-reg/data/test"
SUBJECT_LIST=( "MS2457" )
PLANE_LIST=( "plane0" "plane1" )

# activate environment
module load miniconda3/23.11.0s
source /oscar/rt/9.2/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate roicat
command -v python
python "$EXEC_FILE" --help

# loop through combinations
for SUBJECT in "${SUBJECT_LIST[@]}"; do
for PLANE in "${PLANE_LIST[@]}"; do

    echo ":::::: BEGIN ($SUBJECT, $PLANE) ::::::"
        
    python "$EXEC_FILE" \
        --root-datadir "$ROOT_DATADIR" \
        --subject "$SUBJECT" \
        --plane "$PLANE"
        
    echo ":::::: DONE ($SUBJECT, $PLANE) ::::::"
    echo "--------------------------------------"
    echo ""
    
done
done

