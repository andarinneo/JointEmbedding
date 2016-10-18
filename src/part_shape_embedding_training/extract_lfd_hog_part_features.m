addpath('../');
global_variables;
addpath(genpath(g_piotr_toolbox_path));

part_path = [g_data_folder '/PartAnnotation']; % This needs to go into the global variables definition
n_parts = 4; % 4 parts for chairs
part_list = {'arm', 'back', 'leg', 'seat'};
part_colors = [0 255 0;
               255 0 0;
               0 0 255;
               0 0 0 ];


%% Collect the parts of each object

% t_begin = clock;
% fprintf('Collecting LFD images of shapes listed in \"%s\"...\n', g_shape_list_file);
% view_image_lists = cell(g_lfd_view_num, 1);
% shape_list_fid = fopen(g_shape_list_file);
% line = fgetl(shape_list_fid);
% 
% shape_count = 0;
% while ischar(line)
%     shape_count = shape_count + 1;
%     shape_property = strsplit(line, ' ');
%     
%     model_points_file = fullfile(part_path, shape_property{1}, 'points', [shape_property{2} '.pts']);
%     points3D = importdata(model_points_file, ' ', 0);
%     for i = 1:n_parts
%         part_points_file{i} = fullfile(part_path, shape_property{1}, 'points_label', part_list{i}, [shape_property{2} '.seg']);
%         try % Not all objects have all parts
%             part_labels{i} = importdata(part_points_file{i}, ' ', 0);
%         catch
%             part_labels{i} = zeros(size(points3D,1),1);
%         end
%     end
% 
% %     color_vect = ['g.'; 'r.'; 'b.'; 'y.'; 'k.'];
% %     for i = 1:n_parts
% %         plot3(points3D(:,1).*part_labels{i}, points3D(:,2).*part_labels{i}, points3D(:,3).*part_labels{i}, color_vect(i,:) ); axis equal; hold on;
% %     end
% %     hold off; pause;
% 
%     line = fgetl(shape_list_fid);
% end
% fclose(shape_list_fid);
% t_end = clock;
% fprintf('done (%d shapes, %f seconds)!\n', shape_count, etime(t_end, t_begin));


%% Collect LFD images according to shape list

% t_begin = clock;
% fprintf('Collecting LFD images of shapes listed in \"%s\"...', g_shape_list_file);
% view_image_lists = cell(g_lfd_view_num, 1);
% view_parts_image_lists = cell(g_lfd_view_num, 1);
% shape_list_fid = fopen(g_shape_list_file);
% shape_count = 0;
% while ischar(line)
%     try
%         shape_count = shape_count + 1;
%         
%         line = fgetl(shape_list_fid);
%         shape_property = strsplit(line, ' ');
% 
%         lfd_images_folder = fullfile(g_lfd_images_cropped_folder, shape_property{1}, shape_property{2});
% 
%         lfd_images_cell = extractfield(dir(fullfile(lfd_images_folder, [shape_property{1} '*.png'])), 'name');
%         lfd_images_sorted = sort(lfd_images_cell);
% 
%         lfd_parts_images_cell = extractfield(dir(fullfile(lfd_images_folder, ['parts_' shape_property{1} '*.png'])), 'name');
%         lfd_parts_images_sorted = sort(lfd_parts_images_cell);
% 
%         for i = 1:g_lfd_view_num
%             view_image_lists{i, shape_count} = fullfile(lfd_images_folder, lfd_images_sorted{i});
%             view_parts_image_lists{i, shape_count} = fullfile(lfd_images_folder, lfd_parts_images_sorted{i});
%         end
%     catch exception
%         shape_count = shape_count - 1;
%     end
% end
% fclose(shape_list_fid);
% t_end = clock;
% fprintf('done (%d shapes, %f seconds)!\n', shape_count, etime(t_end, t_begin));


load('borrar.mat');


%% Create part masks

tolerance = 10; % tolerance/255 as color tolerance 
for i = 1:g_lfd_view_num
    lfd_image = double(imread(view_parts_image_lists{1}, 'BackgroundColor', [1 1 1]));
    part_image = double(imread(view_parts_image_lists{1}, 'BackgroundColor', [1 1 1]));
    
    aux = zeros(size(part_image));
    aux(:,:,1) = abs(255 - part_image(:,:,1)) < tolerance;
    aux(:,:,2) = abs(255 - part_image(:,:,2)) < tolerance;
    aux(:,:,3) = abs(255 - part_image(:,:,3)) < tolerance;
    background_mask = (aux(:,:,1) .* aux(:,:,2) .* aux(:,:,3));
    
    for j = 2:n_parts
        segmented_image = double(zeros(size(part_image))*255);
        segmented_image(:,:,1) = segmented_image(:,:,1) + background_mask(:,:,1);
        segmented_image(:,:,2) = segmented_image(:,:,2) + background_mask(:,:,2);
        segmented_image(:,:,3) = segmented_image(:,:,3) + background_mask(:,:,3);
        
        aux = zeros(size(part_image));
        aux(:,:,1) = abs(part_colors(j,1) - part_image(:,:,1)) < tolerance;
        aux(:,:,2) = abs(part_colors(j,2) - part_image(:,:,2)) < tolerance;
        aux(:,:,3) = abs(part_colors(j,3) - part_image(:,:,3)) < tolerance;
        
        mask = aux(:,:,1) .* aux(:,:,2) .* aux(:,:,3);
        
        close all; imshow(mask);
        %final_mask = uint8((1 - (background_mask + mask)) * 255);
        mask = double((1-mask)*255);
        
        segmented_image(:,:,1) = segmented_image(:,:,1) + lfd_image(:,:,1) .* mask;
        segmented_image(:,:,2) = segmented_image(:,:,2) + lfd_image(:,:,2) .* mask;
        segmented_image(:,:,3) = segmented_image(:,:,3) + lfd_image(:,:,3) .* mask;
        
        close all; imshow(segmented_image);
        
        imwrite(segmented_image, [view_parts_image_lists{1} '_part' num2str(j) '.png']);
    end
end


%% Try to get dimension of the HoG feature

hog_dimension = numel(extract_pyramid_hog(view_image_lists{1, 1}, g_lfd_hog_image_size));
fprintf('Each shape will be converted into HoG -PART- feature of %d dimensions!\n', hog_dimension);


%% Compute the HoG feature for the LFD images

fprintf('Start LFD HoG -PART- feature extraction at time %s, it takes for a while!!\n', datestr(now, 'HH:MM:SS'));
local_cluster = parcluster('local');
%poolobj=parpool('local', min(g_lfd_hog_extraction_thread_num, local_cluster.NumWorkers));
report_num = 80; 
report_step = floor((shape_count+report_num-1)/report_num);
t_begin = clock;
view_hog_features = cell(g_lfd_view_num, 1);
for i = 1:g_lfd_view_num
    fprintf('Extracting HoG -PART- feature from LFD images of view %d of %d...\n', i, g_lfd_view_num);
    fprintf([repmat('.', 1, report_num) '\n\n']);
    view_hog_feature = zeros(shape_count, hog_dimension);
    %parfor j = 1:shape_count
    for j = 1:shape_count
        view_hog_feature(j, :) = extract_pyramid_hog(view_image_lists{i, j}, g_lfd_hog_image_size);
        if mod(j, report_step) == 0
            fprintf('\b|\n');
        end
    end    
    view_hog_features{i} = view_hog_feature;
end
%delete(poolobj);
t_end = clock;
fprintf('%f seconds spent on LFD HoG -PART- feature extraction!\n', etime(t_end, t_begin));


%% Save HoG features

t_begin = clock;
fprintf('Save HoG features to \"%s\"...', g_lfd_hog_features_file);
lfd_hog_features = cat(2, view_hog_features{:});
clearvars view_hog_features;
save(g_lfd_hog_features_file, 'lfd_hog_features', '-v7.3');
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));

%exit; % Close MATLAB




