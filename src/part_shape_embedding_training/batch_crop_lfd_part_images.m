addpath('../');
global_variables;


%% Collect LFD images according to shape list

n_classes = max(size(g_shapenet_synset_set));

for iterator = 1:n_classes
    image_list = collect_image_part_list(g_lfd_images_folder, g_shape_list_file);

    local_cluster = parcluster('local');
    poolobj = parpool('local', min(g_lfd_cropping_thread_num, local_cluster.NumWorkers));
    fprintf('Batch cropping LFD images from \"%s\" to \"%s\" ...\n', g_lfd_images_folder, g_lfd_images_cropped_folder);
    batch_crop_parts(g_lfd_images_folder, g_lfd_images_cropped_folder, 0, image_list);
    delete(poolobj);
end

exit;


