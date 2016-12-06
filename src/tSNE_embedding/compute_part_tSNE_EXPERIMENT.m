clearvars; close all;
addpath('../');
global_variables;

part_path = [g_data_folder filesep 'PartAnnotation'];
filename = [part_path filesep 'part_shape_indexes.mat'];
load(filename, 'part_shape_indexes');


%% Load data

load '/home/adrian/JointEmbedding/datasets/shape_embedding/part_shape_embedding_space_03001627.mat'
n_parts = size(part_shape_embedding_space, 2);


%% Load shape list

shape_list_fid = fopen(g_shape_list_file);
line = fgetl(shape_list_fid);
shape_count = 0;
while ischar(line)
    shape_count = shape_count + 1;
    shape_property = strsplit(line, ' ');
    
    image_file = [shape_property{1} '_' shape_property{2} '_a054_e020_t000_d003.png'];
    shape_png_list{shape_count} = fullfile(g_data_folder, 'shape_embedding', 'lfd_images_cropped', shape_property{1}, shape_property{2}, image_file);
    
    line = fgetl(shape_list_fid);
end


%% Run tSNE for every part and store the results

for part = 1:g_n_parts
    n_shapes = size(part_shape_embedding_space{part}, 1);
    
    %% Create 2D embedding space
    
    % Set parameters
    no_dims = 2;
    initial_dims = size(part_shape_embedding_space{part}, 2);
    perplexity = 40; % Originally 30, does not work for the legs, used 50
    
    % Run t-SNE
    mappedX = tsne(part_shape_embedding_space{part}, [], no_dims, initial_dims, perplexity);
    
    % Scale to a 1.0x1.0 square
    scaleX = max(mappedX(:,1)) - min(mappedX(:,1));
    scaleY = max(mappedX(:,2)) - min(mappedX(:,2));
    
    auxX = [];
    auxX(:,1) = ((mappedX(:,1)) - min(mappedX(:,1))) / scaleX;
    auxX(:,2) = ((mappedX(:,2)) - min(mappedX(:,2))) / scaleY;
    
    
    %% Create an embedding image
    
    S = 5000; % size of full embedding image
    G = zeros(S, S, 3, 'uint8');
    s = 50; % size of every single image
    
    for shape = 1:n_shapes
        if mod(shape, 100)==0
            fprintf('%d/%d...\n', shape, n_shapes);
        end
        
        % Location
        a = ceil(auxX(shape,1) * (S-s)+1);
        b = ceil(auxX(shape,2) * (S-s)+1);
        a = a-mod(a-1,s)+1;
        b = b-mod(b-1,s)+1;
        
        if G(a,b,1) ~= 0
            continue % spot already filled
        end
        
        index = part_shape_indexes{part}{shape,1};
        image = imread(shape_png_list{index});
        I = imresize(image,'OutputSize',[s s]);
        
        if size(I,3)==1, I = cat(3,I,I,I); end
        I = imresize(I, [s, s]);
        
        G(a:a+s-1, b:b+s-1, :) = I;
    end
    
    
    %% Save image
    
    imwrite(G, ['results/' 'part_id' int2str(part) '_manifold_view.jpg'], 'jpg');
end


