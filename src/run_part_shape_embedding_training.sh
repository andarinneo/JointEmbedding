#!/bin/bash

first=
last=

usage() { echo "run shape embedding training pipeline from first to last step specified by -f and -l option."; }

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
  echo "-l option is not presented, run the pipeline until last=5!"
  last=8;
fi


## Step 01
## Process the meshes into parts
#if [ "$first" -le 1 ] && [ "$last" -ge 1 ]; then
#  cd part_shape_embedding_training;
#  python3 ./process_geometry_shape_list.py;
#  cd ..;
#fi

## Step 02
## Reduce the meshes to N faces
#if [ "$first" -le 2 ] && [ "$last" -ge 2 ]; then
#  cd part_shape_embedding_training;
#  python3 ./reduce_geometry_shape_list.py;
#  cd ..;
#fi

## Step 03
## Color the meshes based on the labelling
#if [ "$first" -le 3 ] && [ "$last" -ge 3 ]; then
#  cd part_shape_embedding_training;
#  python ./paint_mesh_faces_shape_list.py;
#  cd ..;
#fi

## Step 04
## Generate LFD images
#if [ "$first" -le 4 ] && [ "$last" -ge 4 ]; then
#  cd shape_embedding_training;
#  python3 ./render_lfd_shape_list.py;
#  cd ..;
#fi
## Generate Color labels for LFD images
#if [ "$first" -le 4 ] && [ "$last" -ge 4 ]; then
#  cd part_shape_embedding_training;
#  python3 ./render_lfd_part_shape_list.py;
#  cd ..;
#fi

## Step 05
## Crop LFD images
#if [ "$first" -le 5 ] && [ "$last" -ge 5 ]; then
#  python ./convert_global_variables.py;
#  cd part_shape_embedding_training;
#  /usr/local/MATLAB/R2016a/bin/glnxa64/MATLAB -nodisplay -r batch_crop_lfd_part_images;
#  cd ..;
#fi

## Step 06
## Compute LFD HoG Features
#if [ "$first" -le 6 ] && [ "$last" -ge 6 ]; then
#  python ./convert_global_variables.py;
#  cd part_shape_embedding_training;
#  /usr/local/MATLAB/R2016a/bin/glnxa64/MATLAB -nodisplay -r extract_lfd_hog_part_features;
#  cd ..;
#fi

## Step 07
## Compute shape distance matrix
#if [ "$first" -le 7 ] && [ "$last" -ge 7 ]; then
#  python ./convert_global_variables.py;
#  cd part_shape_embedding_training;
#  /usr/local/MATLAB/R2016a/bin/glnxa64/MATLAB -nodisplay -r compute_part_only_shape_distance_matrix;
#  cd ..;
#fi

## Step 08
## Compute part shape embedding space
#if [ "$first" -le 8 ] && [ "$last" -ge 8 ]; then
#  python ./convert_global_variables.py;
#  cd part_shape_embedding_training;
#  #/usr/local/MATLAB/R2016a/bin/glnxa64/MATLAB -nodisplay -r compute_part_only_shape_embedding_space;
#  cd ..;
#fi
