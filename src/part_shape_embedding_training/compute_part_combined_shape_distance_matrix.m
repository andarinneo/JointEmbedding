clearvars;
addpath('../');
global_variables;

%% Load HoG features

t_begin = clock;
fprintf('Loading HoG features from \"%s\"...\n', g_lfd_hog_part_features_file);
load(g_lfd_hog_part_features_file);
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));


%% Compute distance matrix

t_begin = clock;
fprintf('Computing part shape distance matrix, it takes for a while...\n');

hog_dimension = size(lfd_hog_part_features,2) / ((g_n_parts+1)*g_lfd_view_num);
n_shapes = size(lfd_hog_part_features,1);

whole_shape_lfd = lfd_hog_part_features(:, (1:hog_dimension)); % in this case offset=0, we take the HoG for the whole shape

for part = 1:g_n_parts
    offset = hog_dimension * ((part+1) - 1); % part + 1 because part 1 starts after the whole shape, the -1 goes just to note that accessing matrix positions in a vector requires a minus 1
    part_lfd = lfd_hog_part_features(:, (1:hog_dimension)+offset);
    
    concat_lfd = [whole_shape_lfd part_lfd];
    
    % We save a reduced version of the distance matrix (no zero diagonal or repeated values due to symmetry)
    part_shape_distance_matrix{part} = pdist(concat_lfd);
end

t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));


%% Save shape distance matrix

t_begin = clock;
fprintf('Save part shape distance matrix to \"%s\"...\n', g_part_shape_distance_matrix_file_mat);
save(g_part_shape_distance_matrix_file_mat, 'part_shape_distance_matrix', '-v7.3');

for part = 1:g_n_parts
    part_shape_distance_matrix_NxN{part} = squareform(part_shape_distance_matrix{part});
end
dlmwrite(g_part_shape_distance_matrix_file_txt, part_shape_distance_matrix_NxN, 'delimiter', ' ');

t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));

% exit;
