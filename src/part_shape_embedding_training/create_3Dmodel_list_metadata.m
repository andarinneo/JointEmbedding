clearvars;
addpath('../');
global_variables;

part_path = [g_data_folder filesep 'PartAnnotation'];


%% Begin

t_begin = clock;

fprintf('Collecting a list of 3D models that contains a specific part \"%s\"...\n', g_shape_list_file);

n_parts = size(g_part_labels, 2);

shape_list_fid = fopen(g_shape_list_file);
line = fgetl(shape_list_fid);
shape_count = 0;
while ischar(line)
    shape_count = shape_count + 1;
    
    shape_property = strsplit(line, ' ');
    class_id = shape_property{1};
    model_id = shape_property{2};
    
    %% For each 3D model obtain the part data
    
    points3DFile = [part_path filesep class_id filesep 'points' filesep model_id '.pts'];
    points3DFile_fid = fopen(points3DFile);
    points3D = fscanf(points3DFile_fid, '%f', [3,inf]);
    n_points3D = size(points3D, 2);
    
    for part = 1:n_parts
        partPointFile = [part_path filesep class_id filesep 'points_label' filesep g_part_labels{part} filesep model_id '.seg'];
        partPointFile_fid = fopen(partPointFile);

        if partPointFile_fid == -1
            pointLabels{part} = zeros(n_points3D,1);
        else
            pointLabels{part} = fscanf(partPointFile_fid,'%d');
            fclose(partPointFile_fid);
        end
    end

    points3Dlabels = zeros(n_points3D,1);
    for part = 1:n_parts
        points3Dlabels = points3Dlabels + part * pointLabels{part};
        
        part_presence{part} = sum(pointLabels{part});
    end

    fclose(points3DFile_fid);
    
    model_part_metadata{shape_count} = {class_id, model_id, part_presence};
    
    line = fgetl(shape_list_fid);
end
fclose(shape_list_fid);


%% Create part indexes

for part = 1:n_parts
    indexes = [];
    for shape = 1:shape_count
        if (model_part_metadata{shape}{3}{part} > 0)
            % We add the index and the actual shape id to keep track in case of an index mistake
            cell_value = {shape model_part_metadata{shape}{1} model_part_metadata{shape}{2}};
            indexes = [indexes; cell_value];
        end
    end
    part_shape_indexes{part} = indexes;
end


%% Save metadata and part indexes

filename = [part_path filesep 'part_metadata.mat'];
save(filename, 'model_part_metadata');

filename = [part_path filesep 'part_shape_indexes.mat'];
save(filename, 'part_shape_indexes');

t_end = clock;
fprintf('done (%d shapes, %f seconds)!\n', shape_count, etime(t_end, t_begin));



