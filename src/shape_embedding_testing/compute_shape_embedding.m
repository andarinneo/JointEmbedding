addpath('../');
global_variables;
addpath(genpath(g_minfunc_2012_path));


g_shape_embedding_space_file_mat_gt = '/home/adrian/JointEmbedding/datasets/shape_embedding/shape_embedding_space_03001627(gt).mat';
g_shape_embedding_space_file_mat_myTraining = '/home/adrian/JointEmbedding/datasets/shape_embedding/shape_embedding_space_03001627(meTrained).mat';


%% Load shape distance matrix

t_begin = clock;
fprintf('Loading shape distance matrix from \"%s\"...\n', g_shape_distance_matrix_file_mat);
load(g_shape_distance_matrix_file_mat);
shape_distance_matrix_squared = squareform(shape_distance_matrix);
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));


%% Load GT shape embedding space

t_begin = clock;
fprintf('Loading shape embedding space from \"%s\"...\n', g_shape_embedding_space_file_mat_myTraining);
load(g_shape_embedding_space_file_mat);
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));


%% Load MY shape embedding space

t_begin = clock;
fprintf('Loading shape embedding space from \"%s\"...\n', g_shape_embedding_space_file_mat_gt);
load(g_shape_embedding_space_file_mat);
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));


%% Setting up query_shape_to_embedding_shape_distances

% query_shape_to_embedding_shape_distances is a NxM distance matrix between
% the N query shapes and the M shapes that are already in the shape
% embedding space. You should compute this distance matrix with HoG
% features, or something does similar job.

% Here we just use some shape that are already in the embedding space as 
% query shape to illustrate the usage of the function.
query_shapes = 1:20; 
query_shape_distances_to_embedding_shapes = shape_distance_matrix_squared(query_shapes, :);


%% Setting up query_shape_distance_matrix

% query_shape_distance_matrix is a NxN distance matrix between the N query
% shapes. Similarly, you should compute this distance matrix with HoG
% features, or something that does a similar job.
query_shape_distance_matrix = shape_distance_matrix_squared(query_shapes, query_shapes);


%% Estimate embedding

params.type = 'sammon';
query_embedding_points = GPS(query_shape_distances_to_embedding_shapes, query_shape_distance_matrix, shape_embedding_space, params);



