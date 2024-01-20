function [preprocessDataMatFileFullPath, Xdata, Ydata] = mimic_save_data(matFileFullPath, Xdata, Ydata, IDdata, numCells, index)
% MIMIC_SAVE_DATA Save concatenated data to matlab file
% 
% Syntax: [preprocessDataMatFileFullPath, Xdata, Ydata] = mimic_save_data(matFileFullPath, Xdata, YData, IDdata, numCells)
% 
% Inputs: matFileFullPath -- input .mat file full path including path and filename 
%         Xdata -- output features from mimic_preprocess_2023
%         YData -- output target variables of BP
%         IDdata -- unique identifer for different rows of data
%         points(part_1_1, part_1_i, ... , part_1_1000) = part 1 cell i
%         numCells -- number of cells processed, added to file path
%         
featuresFileName= sprintf('mimic_features_part_%d_numCells_', index);


featuresMatFileFullPath = join([featuresFileName, numCells, ".mat"]);

%featuresMatFileFullPath = fullfile(filePath, [fileNameNoExt, sectionStartEndTimesStr, '_features.mat']); 

% Save features data to the feature output .mat file
save(featuresMatFileFullPath, 'Xdata', 'Ydata', 'IDdata');
fprintf('\nSaved features data to: %s\n', featuresMatFileFullPath);

