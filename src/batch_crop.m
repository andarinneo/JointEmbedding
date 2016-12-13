function batch_crop(src_folder, dst_folder, jitter, src_image_list, cropRatios)

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
    src_image_file = src_image_list{i};
    try
        [I, ~, alpha] = imread(src_image_file);
        
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
    catch
        fprintf('\nFailed to read %s', src_image_file);
    end
    
    if mod(i, report_step) == 0
        fprintf('\b|');
    end
end
fprintf('\n');

if (image_num > 0)
    top = cropBox_mat(1,1);
    bottom = cropBox_mat(1,2);
    left = cropBox_mat(1,3);
    right = cropBox_mat(1,4);
    
    % Save the bounding boxes to reproduce 3D reprojections
    aux_path = strrep(src_image_list{1}, src_folder, '');
    split_path = strsplit(aux_path,'/');
    save([dst_folder '/' split_path{2} '/' 'crop_bbox.mat'], 'top', 'bottom', 'left', 'right');
end


t_end = clock;
fprintf('%f seconds spent on cropping!\n', etime(t_end, t_begin));
end
