clear all;
addpath('../');
global_variables;
addpath(genpath(g_piotr_toolbox_path));

part_path = [g_data_folder '/PartAnnotation']; % This needs to go into the global variables definition
part_list = {'arm', 'back', 'leg', 'seat'};
part_colors = [0 255 0;
               255 0 0;
               0 0 255;
               0 0 0 ];  % ONLY for chairs


%% Collect LFD images according to shape list

t_begin = clock;
fprintf('Collecting LFD images of shapes listed in \"%s\"...', g_shape_list_file);
view_image_lists = cell(g_lfd_view_num, 1);
view_parts_image_lists = cell(g_lfd_view_num, 1);
shape_list_fid = fopen(g_shape_list_file);
shape_count = 0;

line = fgetl(shape_list_fid);
while ischar(line)
    try
        shape_count = shape_count + 1;
        shape_property = strsplit(line, ' ');

        lfd_images_folder = fullfile(g_lfd_images_cropped_folder, shape_property{1}, shape_property{2});

        lfd_images_cell = extractfield(dir(fullfile(lfd_images_folder, [shape_property{1} '*.png'])), 'name');
        lfd_images_sorted = sort(lfd_images_cell);

        lfd_parts_images_cell = extractfield(dir(fullfile(lfd_images_folder, ['parts_' shape_property{1} '*.png'])), 'name');
        lfd_parts_images_sorted = sort(lfd_parts_images_cell);

        for view = 1:g_lfd_view_num
            view_image_lists{view, shape_count} = fullfile(lfd_images_folder, lfd_images_sorted{view});
            view_parts_image_lists{view, shape_count} = fullfile(lfd_images_folder, lfd_parts_images_sorted{view});
        end
    catch exception
        shape_count = shape_count - 1;
    end
    
    line = fgetl(shape_list_fid);
end
fclose(shape_list_fid);
t_end = clock;
fprintf('done (%d shapes, %f seconds)!\n', shape_count, etime(t_end, t_begin));


%% Create part masks

% local_cluster = parcluster('local');
% poolobj = parpool('local', min(g_lfd_hog_extraction_thread_num, local_cluster.NumWorkers));
% 
% tolerance = 1; % color tolerance 
% t_begin = clock;
% 
% % for shape = 1:shape_count
% parfor shape = 1:shape_count
% 
%     for view = 1:g_lfd_view_num
%         lfd_image = double(imread(view_image_lists{view, shape}, 'BackgroundColor', [1 1 1]));
%         part_image = double(imread(view_parts_image_lists{view, shape}, 'BackgroundColor', [1 1 1]));
% 
%         aux = zeros(size(part_image));
%         aux(:,:,1) = abs(255 - part_image(:,:,1)) < tolerance;
%         aux(:,:,2) = abs(255 - part_image(:,:,2)) < tolerance;
%         aux(:,:,3) = abs(255 - part_image(:,:,3)) < tolerance;
%         background_mask = (aux(:,:,1) .* aux(:,:,2) .* aux(:,:,3));
% 
%         for part = 1:g_n_parts
%             segmented_image = double(ones(size(part_image))*255);
%             segmented_image(:,:,1) = segmented_image(:,:,1) + background_mask;
%             segmented_image(:,:,2) = segmented_image(:,:,2) + background_mask;
%             segmented_image(:,:,3) = segmented_image(:,:,3) + background_mask;
% 
%             aux = zeros(size(part_image));
%             aux(:,:,1) = abs(part_colors(part,1) - part_image(:,:,1)) < tolerance;
%             aux(:,:,2) = abs(part_colors(part,2) - part_image(:,:,2)) < tolerance;
%             aux(:,:,3) = abs(part_colors(part,3) - part_image(:,:,3)) < tolerance;
% 
%             mask = aux(:,:,1) .* aux(:,:,2) .* aux(:,:,3);
%             if (sum(mask(:)) > 0)
%                 se = strel('sphere',2);
%                 dilated_mask = imclose(mask,se);
%                 dilated_mask = dilated_mask/max(dilated_mask(:));
%             else
%                 dilated_mask = mask;
%             end
%             
% 
%             inverse_lfd_image = (255 - lfd_image);
% 
%             segmented_image(:,:,1) = segmented_image(:,:,1) - (inverse_lfd_image(:,:,1) .* dilated_mask);
%             segmented_image(:,:,2) = segmented_image(:,:,2) - (inverse_lfd_image(:,:,2) .* dilated_mask);
%             segmented_image(:,:,3) = segmented_image(:,:,3) - (inverse_lfd_image(:,:,3) .* dilated_mask);
%             segmented_image = segmented_image/255;
% 
%     %         close all;
%     %         subplot(1,4,1); imshow(mask);
%     %         subplot(1,4,2); imshow(dilated_mask);
%     %         subplot(1,4,3); imshow(segmented_image);
%     %         subplot(1,4,4); imshow(lfd_image/255);
% 
%             [dst_folder, filename, extension] = fileparts(view_image_lists{view, shape});
%             mod_filename = ['part' num2str(part) '_' filename];
%             full_name = fullfile(dst_folder, [mod_filename extension]);
% 
%             imwrite(segmented_image, full_name, 'Alpha', dilated_mask);
%         end
%     end
% end
% delete(poolobj);
% 
% t_end = clock;
% fprintf('%f seconds spent on LFD -PART- segmentation!\n', etime(t_end, t_begin));


%% Try to get dimension of the HoG feature

hog_dimension = numel(extract_pyramid_hog(view_image_lists{1, 1}, g_lfd_hog_image_size)); % the HoG descriptor
part_hog_dimension = hog_dimension * (g_n_parts+1); % The full size based on the fact that it keeps the whole shape + part descriptors
fprintf('Each shape will be converted into HoG -PART- feature of %d dimensions!\n', hog_dimension);


%% Compute the HoG feature for the LFD images

fprintf('Start LFD HoG -PART- feature extraction at time %s, it takes for a while!!\n', datestr(now, 'HH:MM:SS'));

% local_cluster = parcluster('local');
% poolobj=parpool('local', min(g_lfd_hog_extraction_thread_num, local_cluster.NumWorkers));

report_num = 80; 
report_step = floor((shape_count+report_num-1)/report_num);
t_begin = clock;
view_hog_part_features = cell(g_lfd_view_num, 1);
for view = 1:g_lfd_view_num
    fprintf('Extracting HoG -PART- feature from LFD images of view %d of %d...\n', view, g_lfd_view_num);
    fprintf([repmat('.', 1, report_num) '\n\n']);
    view_hog_part_feature = zeros(shape_count, part_hog_dimension);
    
%     parfor shape = 1:shape_count
    for shape = 1:2 % shape_count
        aux_hog_part_feature = extract_pyramid_hog(view_image_lists{view, shape}, g_lfd_hog_image_size);
        
        for part = 1:g_n_parts
            [dst_folder, filename, extension] = fileparts(view_image_lists{view, shape});
            mod_filename = ['part' num2str(part) '_' filename];
            full_name = fullfile(dst_folder, [mod_filename extension]);
            
            aux_hog_part_feature = [aux_hog_part_feature extract_pyramid_hog(full_name, g_lfd_hog_image_size)];
        end
        
        view_hog_part_feature(shape, :) = aux_hog_part_feature;
        
        if mod(shape, report_step) == 0
            fprintf('\b|\n');
        end
    end
    
    view_hog_part_features{view} = view_hog_part_feature;
end
% delete(poolobj);
t_end = clock;
fprintf('%f seconds spent on LFD HoG -PART- feature extraction!\n', etime(t_end, t_begin));


%% Save HoG features

t_begin = clock;
fprintf('Save HoG part features to \"%s\"...\n', g_lfd_hog_part_features_file);
lfd_hog_part_features = cat(2, view_hog_part_features{:});
% clearvars view_hog_part_features;
% save(g_lfd_hog_part_features_file, 'lfd_hog_part_features', '-v7.3');
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));

% exit; % Close MATLAB




