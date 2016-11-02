
v = 1;
s = 1;
p = 2 + 1;

aux = 0;

for v=1:g_lfd_view_num
    for s=1:2
        for p=1:(g_n_parts+1)

%% Test the HoG function
%             [dst_folder, filename, extension] = fileparts(view_image_lists{v, s});
%             mod_filename = ['part' num2str(p-1) '_' filename];
%             full_name = fullfile(dst_folder, [mod_filename extension]);
%             
%             aux_hog_part_feature = extract_pyramid_hog(full_name, g_lfd_hog_image_size);
%             full_image_hog_part_feature = extract_pyramid_hog(view_image_lists{v, s}, g_lfd_hog_image_size);
%% END Test the HoG function
     
            hog_desc_v1_s2_p1 = view_hog_part_features{v}(s,(1:hog_dimension)+hog_dimension*(p-1));

            offset = hog_dimension * (((g_n_parts+1) * (v-1) + p) - 1);
            vec_hog_desc_v1_s2_p1 = lfd_hog_part_features(s,(1:hog_dimension)+offset);

            aux = aux + sum(hog_desc_v1_s2_p1 - vec_hog_desc_v1_s2_p1);
        end
    end
end

disp(aux);
% aux_hog_part_feature([1:10]+10000)
% full_image_hog_part_feature([1:10]+10000)