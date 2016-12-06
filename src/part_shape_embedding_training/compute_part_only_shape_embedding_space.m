clearvars; close all;
addpath('../');
global_variables;

part_path = [g_data_folder filesep 'PartAnnotation'];
filename = [part_path filesep 'part_metadata.mat'];
load(filename, 'model_part_metadata');


%% Create indexes for each of the part manifolds

n_shapes = size(model_part_metadata,2);

part_existence_bool = false(n_shapes, g_n_parts);
for shape = 1:n_shapes
    part_existence_bool(shape,:) = [model_part_metadata{shape}{3}{:}];
end

for part = 1:g_n_parts
    part_indexes{part} = find(part_existence_bool(:,part));
end


%% Load part shape distance matrix

t_begin = clock;
fprintf('Loading part shape distance matrix from \"%s\"...\n', g_part_shape_distance_matrix_file_mat);
load(g_part_shape_distance_matrix_file_mat);
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));


%% Create new distance matrices based on the existence of parts

for part = 1:g_n_parts
    part_shape_distance_matrix_NxN = squareform(part_shape_distance_matrix{part});
    
    cropped_part_shape_distance_matrix_NxN = part_shape_distance_matrix_NxN(part_indexes{part}, part_indexes{part});
    
    cropped_part_shape_distance_matrix{part} = squareform(cropped_part_shape_distance_matrix_NxN);
end



%% Compute part shape embedding space

t_begin = clock;
fprintf('Computing part shape embedding space, it takes for a while...\n');
options = statset('Display', 'iter', 'MaxIter', 128);

% This 3 values are wrong in part 1 distance matrix(vector form) for chair class: 3121619, 8140026, 9457494
% This is normal because the 3 cases are exactly the same 3D models

% We fix the issue for all classes by adding a small epsilon to those values
epsilon = 0.01; % 0.0001
for part = 1:g_n_parts
    indexes = (cropped_part_shape_distance_matrix{part} == 0);
    n_empty_val = sum(indexes);
    cropped_part_shape_distance_matrix{part}(indexes) = rand(1, n_empty_val) * epsilon;
end

for part = 1:g_n_parts
    [aux_part_shape_embedding_space, stress, disparities] = mdscale(cropped_part_shape_distance_matrix{part}, g_part_shape_embedding_space_dimension, 'criterion', 'sammon', 'options', options);
    
    part_shape_embedding_space{part} = aux_part_shape_embedding_space;
end

t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));


%% Save embedding space

t_begin = clock;
fprintf('Save part shape embedding space to \"%s\"...\n', g_part_shape_embedding_space_file_mat);
save(g_part_shape_embedding_space_file_mat, 'part_shape_embedding_space', '-v7.3');

for part = 1:g_n_parts
    [pathstr, name, ext] = fileparts(g_part_shape_embedding_space_file_txt);
    dlmwrite(fullfile(pathstr, [name '_part' int2str(part) ext]), part_shape_embedding_space{part}, 'delimiter', ' ');
end

t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));

exit;


