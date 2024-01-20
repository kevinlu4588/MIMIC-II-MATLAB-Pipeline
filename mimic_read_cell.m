function mimic_read_cell(matFileFullPath, index, cellNumber)
% mimic_read_matFile Generate SOP data for a 2022/2023 collected single .mat file
%
% Syntax: mimic_read_matFile(matFileFullPath)
% Inputs: matFileFullPath -- input .mat file full path including path and filename
% Because the data is already preprocessed, section start and end removed
% Kevin Lu, October 17 2023
%% Preprocess data, calculate featues, and save preprocessed and features to a .mat file
load(matFileFullPath)
fprintf('\nPreprocessing Kaggle data, calculating features and segmenting data, and saving them for deep learning...\n');

numberCellsToProcess = 1000;
finalXData = [];
finalYData = [];
finalIDData = [];
badSampleIds = [];
i = cellNumber;
        
% Initialize cell arrays to store the data

ppg_cells = cell(1, 0);
bp_cells = cell(1, 0);
ecg_cells = cell(1, 0);

sample = p{1,i};
ppg_sample = sample(1, :);
bp_sample = sample(2, :);
ecg_sample = sample(3, :);

[ppg_rows, ppg_cols] = size(ppg_sample);
ppg_cells = horzcat(ppg_cells, ppg_sample);
bp_cells = horzcat(bp_cells, bp_sample);
ecg_cells = horzcat(ecg_cells, ecg_sample);

%Converts the cell arrays to matlab volumn vectors
ECG = cell2mat(ecg_cells)'; %column vector of ECG From Mimic-II
Finapres = cell2mat(bp_cells)'; %column vector of BP from Mimic-II
NIRS = cell2mat(ppg_cells)'; % Column vector of PPG from Mimic-II


disp(["Sample #: ", i])

%try SOP generation process, if breaks then error is caught
%and sample id is added to bad sample ids list
% try
%Process and extract features from preprocessed data
[preprocessDataMatFileFullPath, XData, YData] = mimic_individual_preprocess_saveFeatures_20222023data(matFileFullPath, ECG, Finapres, NIRS);

%Build IDdata matlab cell array
numEntries = length(XData);
IDData = cell(numEntries, 1);
cell_part = sprintf('part_%d_cell_', index);
IDString = strcat(cell_part, num2str(i));

for j = 1:numEntries

IDData{j} = IDString;
end
finalIDData = [finalIDData; IDData];
finalXData = [finalXData; XData];
finalYData = [finalYData; YData];
%catch bad sample, add to list
% catch
%  disp(["error during sample", i])
%  badSampleIds = [badSampleIds, i];
% end
% Generate the new filename with the specified index
newFilename = sprintf('mimic_part%d_bad_sampleIds.mat', index);

save(newFilename, "badSampleIds")

%Save data, convert the IDData strings from cell array => normal string so
%it can be processed in Python
mimic_save_data(matFileFullPath, finalXData, finalYData, cellstr(finalIDData), cellNumber, index);
disp("finished!")



