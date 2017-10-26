clearvars; close all;
addpath('../');
global_variables;


%% Load data

% load '/home/adrian/JointEmbedding/datasets/shape_embedding/lfd_hog_part_features_03001627_(originalHoG).mat'
load(g_part_shape_embedding_space_file_mat);

n_parts = size(part_shape_embedding_space, 2);
n_shapes = size(part_shape_embedding_space{1}, 1);


%% Load shape list

shape_list_fid = fopen(g_shape_list_file);
line = fgetl(shape_list_fid);
shape_count = 0;
while ischar(line)
    shape_count = shape_count + 1;
    shape_property = strsplit(line, ' ');
    
%     image_file = [shape_property{1} '_' shape_property{2} '_a054_e020_t000_d003.png'];  % lfd image
%     image_file = ['parts_' shape_property{1} '_' shape_property{2} '_a054_e020_t000_d003.png'];  % part segmentation image
%     shape_png_list{shape_count} = fullfile(g_data_folder, 'shape_embedding', 'lfd_images_cropped', shape_property{1}, shape_property{2}, image_file);
    
    image_file = [shape_property{1} '_' shape_property{2} '_a054_e020_t000_d003.png'];  % rendered image
    shape_png_list{shape_count} = fullfile('/home/adrian/Desktop/la_web_con_todos/image_based_Back+Legs_blending_case1_files', image_file);
    
    line = fgetl(shape_list_fid);
end

if shape_count ~= n_shapes
    frpintf('\nThere is a problem in the number of shapes in the manifold...\n\n');
end


%% Load image embedded list

M1 = dlmread('/home/adrian/Desktop/image_embedding_images/img_embeddings_part1.txt');
M2 = dlmread('/home/adrian/Desktop/image_embedding_images/img_embeddings_part2.txt');
M3 = dlmread('/home/adrian/Desktop/image_embedding_images/img_embeddings_part3.txt');
M4 = dlmread('/home/adrian/Desktop/image_embedding_images/img_embeddings_part4.txt');
n_embedded_images = size(M1, 1);

mixed_part_shape_embedding_space{1} = [M1; part_shape_embedding_space{1}];
mixed_part_shape_embedding_space{2} = [M2; part_shape_embedding_space{2}];
mixed_part_shape_embedding_space{3} = [M3; part_shape_embedding_space{3}];
mixed_part_shape_embedding_space{4} = [M4; part_shape_embedding_space{4}];

img_root = '/home/adrian/Desktop/image_embedding_images/';
img_list_fid = fopen([img_root 'img_filelist.txt']);

line = fgetl(img_list_fid);
img_count = 0;
while ischar(line)
    img_count = img_count + 1;
    
    img_file = [img_root line];  % rendered image
    img_png_list{img_count} = img_file;
    
    line = fgetl(img_list_fid);
end

mixed_png_list = [img_png_list shape_png_list];


%% Run tSNE for every part and store the results

for part_id = 1:g_n_parts
    %% Create 2D embedding space
    
    % Set parameters
    n_dims = 2;
    initial_dims = size(mixed_part_shape_embedding_space{part_id}, 2);
    perplexity = 30;
    
    % Run t-SNE
%     mappedX = tsne(mixed_part_shape_embedding_space, [], n_dims, initial_dims, perplexity);
    
    % Run PPCA - [COEFF,SCORE,LATENT,MU,V,S] = ppca(Y,K)
    [COEFF, SCORE, LATENT, MU, V, R] = ppca(mixed_part_shape_embedding_space{part_id}, n_dims);
    mappedX = R.Recon;
    
    % Scale to a 1.0x1.0 square
    scaleX = max(mappedX(:,1)) - min(mappedX(:,1));
    scaleY = max(mappedX(:,2)) - min(mappedX(:,2));
    
    auxX(:,1) = ((mappedX(:,1)) - min(mappedX(:,1))) / scaleX;
    auxX(:,2) = ((mappedX(:,2)) - min(mappedX(:,2))) / scaleY;
    
    
    %% Create an embedding image
    
    S = 2000; % size of full embedding image
    G = ones(S, S, 3, 'uint8')*255;
    s = 200; % size of every single image
    s_aux = round(s*0.6);  % hardcoded ratio of row/cols of images
    
%     subsampled_vec = [502 6682 5397 2253 2403 130:140]+1+n_embedded_images;
%     subsampled_vec = [378 376 360 693 859 926 1059 1248 subsampled_vec];
    
    subsampled_vec = [502 6682 131]+1+n_embedded_images;
    subsampled_vec = [859 1059 subsampled_vec];
    
    for shape=subsampled_vec %1:n_shapes+n_embedded_images
        
        if mod(shape, 100)==0
            fprintf('%d/%d...\n', shape, n_shapes+n_embedded_images);
        end
        
        % location
        a = ceil(auxX(shape,1) * (S-s)+1);
        b = ceil(auxX(shape,2) * (S-s)+1);
        a = a-mod(a-1,s)+1;
        b = b-mod(b-1,s)+1;
        if (uint32(G(a,b,1))+uint32(G(a,b,2))+uint32(G(a,b,3))) ~= (255*3)
            continue % spot already filled
        end
        
        [image, map, alpha] = imread(mixed_png_list{shape});
        I = imresize(image,'OutputSize',[s s]);
        if size(I,3)==1, I = cat(3,I,I,I); end
        I = imresize(I, [s, s_aux]);
        
        if ~isempty(alpha)
            A = imresize(alpha, [s, s_aux]);
            A = abs(double(A)-255);
            A = uint8(cat(3,A,A,A));
            G(a:a+s-1, b:b+s_aux-1, :) = I + A;
        else
            G(a:a+s-1, b:b+s_aux-1, :) = I;
        end
    end
    
    
    %% Save image
    
    imwrite(G, ['tSNE_embedding/results/' 'part_id' int2str(part_id) '_manifold_view.png'], 'png');
end


