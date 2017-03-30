%  ----------------------------------------------------------------------------
%  Create a list of images that have a segmentation under "syn_images_cropped/"
%  ----------------------------------------------------------------------------
close all; clearvars;

addpath('../');
global_variables;


%% Collect synthetic images according to shape list

key = '*seg.png';
output_file = '../datasets/image_embedding/seg_file_list.txt';
[seg_file_list, n_segmentations] = collect_image_list_fcn(g_syn_images_cropped_folder, g_shape_list_file, output_file, key);

key = '*003.jpg';
output_file = '../datasets/image_embedding/image_list.txt';
[image_list, n_images] = collect_image_list_fcn(g_syn_images_bkg_overlaid_folder, g_shape_list_file, output_file, key);

if abs(n_segmentations-n_images) > 0
    disp('There is an error in the generated images...');
end

disp('Training indexes have been generated...');


%% Collect manifold coordinates for the synthetic images


