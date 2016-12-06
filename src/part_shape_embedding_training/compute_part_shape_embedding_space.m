clearvars;
addpath('../');
global_variables;

%% Load part shape distance matrix

t_begin = clock;
fprintf('Loading part shape distance matrix from \"%s\"...\n', g_part_shape_distance_matrix_file_mat);
load(g_part_shape_distance_matrix_file_mat);
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));


%% Compute part shape embedding space

t_begin = clock;
fprintf('Computing part shape embedding space, it takes for a while...\n');
options = statset('Display', 'iter', 'MaxIter', 128);

% This 3 values are wrong in part 1 distance matrix(vector form) for chair class: 3121619, 8140026, 9457494
% This is normal because the 3 cases are exactly the same 3D models

% We fix the issue for all classes by adding a small epsilon to those values
epsilon = 0.01; % 0.0001
for part = 1:g_n_parts
    indexes = (part_shape_distance_matrix{part} == 0);
    n_empty_val = sum(indexes);
    part_shape_distance_matrix{part}(indexes) = rand(1, n_empty_val) * epsilon;
end

for part = 1:g_n_parts
    [aux_part_shape_embedding_space, stress, disparities] = mdscale(part_shape_distance_matrix{part}, g_part_shape_embedding_space_dimension, 'criterion', 'sammon', 'options', options);
    
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


