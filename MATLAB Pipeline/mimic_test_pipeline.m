% test script for SOP data pipeline
% which consists of mimic_read_matFile and
% mimic_preprocess_saveFeatures_2022.mat

% which Generates SOP data for a 2022/2023 MIMIC-II Data
%
% Kevin Lu October 16th 2023

%mimic_read_matfile takes in matfile path

for i = 1:1
    file_name = sprintf('part_%d.mat', i);
    argument = i;
    mimic_read_matFile(file_name, argument);
end


%mimic_read_cell("part_1.mat", 2, 489);