#!/bin/bash

first=
last=

usage() { echo "run image embedding training pipeline from first to last step specified by -f and -l option."; }

first_flag=0
last_flag=0
while getopts f:l:h opt; do
  case $opt in
  f)
    first_flag=1;
    first=$(($OPTARG))
    ;;
  l)
    last_flag=1;
    last=$(($OPTARG))
    ;;
  h)
    usage; exit;;
  esac
done

shift $((OPTIND-1))

if [ $first_flag -eq 0 ]
then
  echo "-f option is not presented, run the pipeline from first=1!"
  first=1;
fi

if [ $last_flag -eq 0 ]
then
  echo "-l option is not presented, run the pipeline until last=8!"
  last=8;
fi


## Step 01
## Download SUN2012 data for background
#if [ "$first" -le 1 ] && [ "$last" -ge 1 ]; then
#  cd image_embedding_training;
#  ./prepare_sun2012_data.py;
#  cd ..;
#fi

## Step 02
## Download caffemodels
#if [ "$first" -le 2 ] && [ "$last" -ge 2 ]; then
#  cd image_embedding_training;
#  ./prepare_caffemodels.py;
#  cd ..;
#fi

## Step 03
## Generate view distribution files
#if [ "$first" -le 3 ] && [ "$last" -ge 3 ]; then
#  cd image_embedding_training;
#  python ./gen_view_distribution_files.py;
#  cd ..;
#fi

# Step 04
# Generate synthetic images
if [ "$first" -le 4 ] && [ "$last" -ge 4 ]; then
  cd part_image_embedding_training;
  python3 ./render_part_syn_shape_list.py;
  cd ..;
fi

## Step 05
## Crop synthetic images
#if [ "$first" -le 5 ] && [ "$last" -ge 5 ]; then
#  python ./convert_global_variables.py;
#  cd image_embedding_training;
#  /usr/local/MATLAB/R2016a/bin/glnxa64/MATLAB -nodisplay -r batch_crop_syn_images;
#  cd ..;
#fi

## Step 06
## Background overlay
#if [ "$first" -le 6 ] && [ "$last" -ge 6 ]; then
#  python ./convert_global_variables.py;
#  cd image_embedding_training;
#  /usr/local/MATLAB/R2016a/bin/glnxa64/MATLAB -nodisplay -r background_overlay_syn_images;
#  cd ..;
#fi

## Step 07
## Generate synthetic images filelist
#if [ "$first" -le 7 ] && [ "$last" -ge 7 ]; then
#  cd image_embedding_training;
#  python ./gen_syn_filelist.py
#  cd ..;
#fi

## Step 08
## Convert images into pool5 lmdb
#if [ "$first" -le 8 ] && [ "$last" -ge 8 ]; then
#  cd image_embedding_training;
#  python ./extract_pool5_feats.py
#  cd ..;
#fi
