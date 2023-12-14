#!/bin/bash

# TODO: consider changing "4-muse" to "4-roicat"

# define constants
SINGLE_SESSION_SUBDIRS=( "0-raw" "1-moco" "2-deepcad" "3-suite2p" "4-muse" )
MULTI_SESSION_DIR="multi-session"
LOG_DIR="./logs"

# define variables
ROOT_DATADIR="/oscar/data/afleisc2/collab/multiday-reg/data/testdir"
SUBJECT_LIST=( "MS2457" )
DATE_LIST=( "20230902" "20230907" "20230912" )
PLANE_LIST=( "plane0" "plane1" )

# make log folders
for SUBDIR in "${SINGLE_SESSION_SUBDIRS[@]}"; do
    mkdir -p "$LOG_DIR/$SUBDIR"
done

mkdir -p "$LOG_DIR/4-roicat"

# loop through combinations
for SUBJECT in "${SUBJECT_LIST[@]}"; do

echo ":::::: BEGIN ($SUBJECT) ::::::"
echo "Creating single-session directories:"

for DATE in "${DATE_LIST[@]}"; do
for PLANE in "${PLANE_LIST[@]}"; do
for SUBDIR in "${SINGLE_SESSION_SUBDIRS[@]}"; do

    SINGLE_DIR="$ROOT_DATADIR/$SUBJECT/$DATE/$SUBDIR/$PLANE"
    mkdir -p "$SINGLE_DIR"
    echo -e "\t$SINGLE_DIR"
    
done
done
done

echo "Creating multi-session directories:"
for PLANE in "${PLANE_LIST[@]}"; do
    MULTI_DIR="$ROOT_DATADIR/$SUBJECT/$MULTI_SESSION_DIR/$PLANE"
    mkdir -p "$MULTI_DIR"
    echo -e "\t$MULTI_DIR"
done


echo ":::::: DONE ($SUBJECT) ::::::"
echo "------------------------------"
echo ""

done

# list raw dirs
echo "Please put raw TIF data in these folders, according to their name"
find "$ROOT_DATADIR" -type d -wholename "**/0-raw/*"

