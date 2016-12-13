function [ image_list ] = collect_image_list( folder, shape_list_file )
%COLLECT_IMAGE_LIST Summary of this function goes here
%   Detailed explanation goes here

t_begin = clock;
fprintf('Collecting synthetic images of shapes listed in \"%s\"...\n', shape_list_file);
shape_list_fid = fopen(shape_list_file);
line = fgetl(shape_list_fid);
image_count = 0;


% 1           2           3           4           5        4158        4159        4160        4161        4162


% load('borrar.mat', 'stupid_adrian');
counter1 = 1;
% counter2 = 1;

while ischar(line)
%     if (counter1 == stupid_adrian(counter2) )
        shape_property = strsplit(line, ' ');
        shape_images_folder = fullfile(folder, shape_property{1}, shape_property{2});
        image_count = image_count + length(dir(fullfile(shape_images_folder, '*.png')));
        
%         counter2 = counter2 + 1;
%     end
    aux_num(counter1) = length(dir(fullfile(shape_images_folder, '*.png')));
    counter1 = counter1 + 1;
    
    line = fgetl(shape_list_fid);
end
fclose(shape_list_fid);


%% Allocate cells

image_list = cell(image_count, 1);
shape_list_fid = fopen(shape_list_file);
line = fgetl(shape_list_fid);
image_count = 0;



% counter1 = 1;
% counter2 = 1;

while ischar(line)
%     if (counter1 == stupid_adrian(counter2) )
        shape_property = strsplit(line, ' ');
        shape_images_folder = fullfile(folder, shape_property{1}, shape_property{2});
        shape_images = extractfield(dir(fullfile(shape_images_folder, '*.png')), 'name');
        shape_image_count = length(shape_images);
        for i = 1:shape_image_count
            image_list{image_count+i} = fullfile(shape_images_folder, shape_images{i});
        end
        image_count = image_count + shape_image_count;
        
%         counter2 = counter2 + 1;
%     end
%     counter1 = counter1 + 1;
    
    line = fgetl(shape_list_fid);
end
fclose(shape_list_fid);
t_end = clock;
fprintf('done (%d images, %f seconds)!\n', image_count, etime(t_end, t_begin));

end

