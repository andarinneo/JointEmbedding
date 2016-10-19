function batch_crop_parts(src_folder, dst_folder, jitter, src_image_list, cropRatios)

if nargin < 5
    cropRatios = zeros(1,4);
end

fprintf('Start croping at time %s...it takes for a while!!\n', datestr(now, 'HH:MM:SS'));
report_num = 80;
fprintf([repmat('.',1,report_num) '\n\n']);
image_num = length(src_image_list);
report_step = floor((image_num+report_num-1)/report_num);

t_begin = clock;

cropBox_mat = zeros(image_num, 4);
%for i = 1:image_num
parfor i = 1:image_num

    %% Crop lfd image
    
    src_image_file = src_image_list{i};
    try
        [I, ~, alpha] = imread(src_image_file);       
    catch
        fprintf('Failed to read %s\n', src_image_file);
    end
    [alpha, top, bottom, left, right] = crop_gray(alpha, 0, jitter, cropRatios);      
    I = I(top:bottom, left:right, :);
    
    cropBox_mat(i,:) = [top bottom left right];

    if numel(I) == 0
        fprintf('Failed to crop %s (empty image after crop)\n', src_image_file);
    else
        dst_image_file = strrep(src_image_file, src_folder, dst_folder);
        [dst_image_file_folder, ~, ~] = fileparts(dst_image_file);
        if ~exist(dst_image_file_folder, 'dir')
            mkdir(dst_image_file_folder);
        end
        imwrite(I, dst_image_file, 'png', 'Alpha', alpha);
    end
    
    
    %% Crop part color image using the same cropBox
    
    [path_folder, file_name, extension] = fileparts(src_image_file);
    src_part_image_file = fullfile(path_folder, ['parts_' file_name extension]);
    try
        [I_parts, ~, ~] = imread(src_part_image_file);       
    catch
        fprintf('Failed to read part file %s\n', src_part_image_file);
    end     
    I_parts = I_parts(top:bottom, left:right, :);
    
    if numel(I_parts) == 0
        fprintf('Failed to crop part image %s (empty image after crop)\n', src_image_file);
    else
        dst_image_file = strrep(src_part_image_file, src_folder, dst_folder);
        [dst_image_file_folder, ~, ~] = fileparts(dst_image_file);
        if ~exist(dst_image_file_folder, 'dir')
            mkdir(dst_image_file_folder);
        end
        imwrite(I_parts, dst_image_file, 'png', 'Alpha', alpha);
    end
    
    
    %% Print iteration
    if mod(i, report_step) == 0
        fprintf('\b|\n');
    end
end   

% Save the bounding boxes to reproduce 3D reprojections
aux_path = strrep(src_image_list{1}, src_folder, '');
split_path = strsplit(aux_path,'/');
%save([dst_folder '/' split_path{2} '/' 'crop_bbox.mat'], 'top', 'bottom', 'left', 'right');

t_end = clock;
fprintf('%f seconds spent on cropping!\n', etime(t_end, t_begin));
end
