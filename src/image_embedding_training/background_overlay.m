function background_overlay(src_folder, dst_folder, src_image_list, bkgFilelist, bkgFolder, clutteredBkgRatio)

sunImageList = importdata(bkgFilelist);

fprintf('Start overlaying images at time %s, it takes for a while...\n', datestr(now, 'HH:MM:SS'));
report_num = 80;
fprintf([repmat('.',1,report_num) '\n\n']);
image_num = length(src_image_list);
report_step = floor((image_num+report_num-1)/report_num);
t_begin = clock;
% for i = 1:length(src_image_list)
parfor i = 1:image_num
    src_image_file = src_image_list{i};
    try
        [I, ~, alpha] = imread(src_image_file);       
        
        s = size(I);
        fh = s(1); fw = s(2);
        mask = double(alpha) / 255;
        mask = repmat(mask,1,1,3);

        if rand() > clutteredBkgRatio
            I = uint8(double(I) .* mask + double(rand()*255) * (1 - mask));
        else
            while true
                id = randi(length(sunImageList));
                bg = imread(fullfile(bkgFolder, sunImageList{id}));
                s = size(bg);
                bh = s(1); bw = s(2);
                if bh < fh || bw < fw
                    %fprintf(1, '.');
                    continue;
                end
                if length(s) < 3
                    continue;
                end
                break;
            end
            by = randi(bh - fh + 1);
            bx = randi(bw - fw + 1);
            bgcrop = bg(by:(by+fh-1), bx:(bx+fw-1), :);

            I = uint8(double(I) .* mask + double(bgcrop) .* (1 - mask));
        end

        if numel(I) == 0
            fprintf('Failed to overlay %s (empty image after crop)\n', src_image_file);
        else
            dst_image_file = strrep(src_image_file, src_folder, dst_folder);
            dst_image_file = strrep(dst_image_file, 'png', 'jpg');
            [dst_image_file_folder, ~, ~] = fileparts(dst_image_file);
            if ~exist(dst_image_file_folder, 'dir')
                mkdir(dst_image_file_folder);
            end
            imwrite(I, dst_image_file, 'jpg');
        end
    catch
        fprintf('Failed to read %s\n', src_image_file);
    end
    
    if mod(i, report_step) == 0
        fprintf('\b|\n');
    end
end
t_end = clock;
fprintf('%f seconds spent on background overlay!\n', etime(t_end, t_begin));
