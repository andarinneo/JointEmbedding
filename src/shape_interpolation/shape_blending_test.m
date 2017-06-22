clearvars;
addpath('../');
global_variables;

%% Load part shape distance matrix

t_begin = clock;
fprintf('Loading part shape distance matrix from \"%s\"...\n', g_part_shape_distance_matrix_file_mat);
load(g_part_shape_distance_matrix_file_mat);
t_end = clock;
fprintf('done (%f seconds)!\n', etime(t_end, t_begin));


%% Create sqaureform distance matrices

part1_manifold_NxN = squareform(part_shape_distance_matrix{1});
part2_manifold_NxN = squareform(part_shape_distance_matrix{2});
part3_manifold_NxN = squareform(part_shape_distance_matrix{3});
part4_manifold_NxN = squareform(part_shape_distance_matrix{4});

n_shapes = size(part1_manifold_NxN,1);


%% Do experiments to measure blending performance
% part ID: 1 armrest, 2 back, 3 legs, 4 seat

% % It works
% md5_a = 183;  % part 2
% dist_to_part_from_a = part2_manifold_NxN(md5_a,:);
% md5_b = 219;  % part 3
% dist_to_part_from_b = part3_manifold_NxN(md5_b,:);

% % It works
% md5_a = 183;  % part 1
% dist_to_part_from_a = part1_manifold_NxN(md5_a,:);
% md5_b = 219;  % part 3
% dist_to_part_from_b = part3_manifold_NxN(md5_b,:);

% % It works
% md5_a = 183;  % part 1
% dist_to_part_from_a = part1_manifold_NxN(md5_a,:);
% md5_b = 3842;  % part 2
% dist_to_part_from_b = part2_manifold_NxN(md5_b,:);

% % It works
% md5_a = 3842;  % part 2
% dist_to_part_from_a = part2_manifold_NxN(md5_a,:);
% md5_b = 219;  % part 3
% dist_to_part_from_b = part3_manifold_NxN(md5_b,:);

% % It works
% md5_a = 4274;  % part 2
% dist_to_part_from_a = part2_manifold_NxN(md5_a,:);
% md5_b = 219;  % part 1
% dist_to_part_from_b = part1_manifold_NxN(md5_b,:);

% % It works
% md5_a = 2679;  % part 3
% dist_to_part_from_a = part3_manifold_NxN(md5_a,:);
% md5_b = 219;  % part 1
% dist_to_part_from_b = part1_manifold_NxN(md5_b,:);

% % It works
% md5_a = 2679;  % part 3
% dist_to_part_from_a = part3_manifold_NxN(md5_a,:);
% md5_b = 262;  % part 2
% dist_to_part_from_b = part2_manifold_NxN(md5_b,:);

% % It works
% md5_a = 2679;  % part 2
% dist_to_part_from_a = part2_manifold_NxN(md5_a,:);
% md5_b = 3694;  % part 3
% dist_to_part_from_b = part3_manifold_NxN(md5_b,:);

% % It works
% md5_a = 2679;  % part 2
% dist_to_part_from_a = part2_manifold_NxN(md5_a,:);
% md5_b = 3694;  % part 3
% dist_to_part_from_b = part3_manifold_NxN(md5_b,:);

% % It works
% md5_a = 4859;  % part 2
% dist_to_part_from_a = part2_manifold_NxN(md5_a,:);
% md5_b = 4839;  % part 1
% dist_to_part_from_b = part1_manifold_NxN(md5_b,:);

% It works
md5_a = 4859;  % part 1
dist_to_part_from_a = part1_manifold_NxN(md5_a,:);
md5_b = 4648;  % part 2
dist_to_part_from_b = part2_manifold_NxN(md5_b,:);



%% Minimize sumed distances

sumed_distances = dist_to_part_from_a + dist_to_part_from_b;
[dist_val, dist_idx] = sort(sumed_distances);

dist_val(1:10)
dist_idx(1:10)


%% Minimize sumed ranking

[val_p2, idx_p2] = sort(dist_to_part_from_a);
[val_p3, idx_p3] = sort(dist_to_part_from_b);

aux_ranking = [1:n_shapes; idx_p2];
[~, b] = sort(aux_ranking(2,:));
ranking_p2 = aux_ranking(1,b);

aux_ranking = [1:n_shapes; idx_p3];
[~, b] = sort(aux_ranking(2,:));
ranking_p3 = aux_ranking(1,b);

sumed_ranking = ranking_p2 + ranking_p3;
[rank_val, rank_idx] = sort(sumed_ranking);

rank_val(1:10)
rank_idx(1:10)









