function [preprocessDataMatFileFullPath, Xdata, Ydata] = mimic_individual_preprocess_saveFeatures_20222023data(matFileFullPath, ECG, Finapres, NIRS)
% PREPROCESS_SAVEFEATURES_20222023DATA Preprocess data (data collected during 2017/2018), calculate features and segment data, and save them to output files for deep learning
% 
% Syntax: [preprocessDataMatFileFullPath, featuresMatFileFullPath, segDataOutFileDir] = preprocess_saveFeatures_20222023data(matFileFullPath)
% 
% Inputs: matFileFullPath -- input .mat file full path including path and filename 
%         
% Outputs: preprocessDataMatFileFullPath -- preprocess data output .mat file full path
%          featuresMatFileFullPath -- features output .mat file full path
%          segDataOutFileDir -- segmented data output directory name
%
% Example Usages: [preprocessDataMatFileFullPath, featuresMatFileFullPath, segDataOutFileDir] = ...
%                      preprocess_saveFeatures_20222023data('C:\temp\09_30_2022\2022-0930-134657-10403.mat');
%                Or:
%                [preprocessDataMatFileFullPath, featuresMatFileFullPath, segDataOutFileDir] = ...
%                      preprocess_saveFeatures_20222023data('C:\temp\09_30_2022\2022-0930-134657-10403.mat');
%
%
% Baseline code majorityly from 1) Dr. Qiao Li at Qiao\code\read_data_for_deep_learning.m, 
% and some from yshi from 2) yshi at:
% calMAD&calPTT_v4.1\read_NIN_SE_Normalization_calMAD_calPTT.m, as well as
% from 3) Dr. Quan Zhang at: NINExperiment\read_NIN_2022.m, and
% from 4) Dr. Ye Yang at: YeCodes\genSOPData_20172018\preprocess_saveFeatures_20172018data.m
% 
% This is based on the code framework in 4) that was based on 1), 2) and 3)
% and with necessary additions and modifications, and with necessary
% additions and modifications in this script for 2022/2023 .nin files
% processing as well.
% 
% See above mentioned baseline code for details
%
% 1) Kevin Lu October 16, 2023 Initial adaption to MIMIC-II dataset
% a) Changed method header(removed section start and end as the data has
% been preprocessed already and there are no manual data collection errors)
% b) Adapted data channels to MIMIC Data Formatting
% bi) Created appropriate ECG, Finapres, and PPG column vectors from
% MIMIC-II part1.mat file
% c) Changed fsh = 125Hz from 250Hz
% d) Removed STAT Calculations + outputs
% 
% 2) Kevin Lu October 23, 2023 Removed some unnecessary preprocessing
% a) Commented out '60Hz_norch_FIR_for_250.mat' ECG filter
% b) Also commented out 60 Hz power line interference noise from Finapres
% signal
% c) Commented out Finapres A/D to mmHg calculation(BP data already in
% mmHg)

% Kevin Lu November 21st
% 3) Sample must have > 4000 samples, but edited 

fsh = 125; %  
ECG = ECG;
Finapres = Finapres;
NIRS = NIRS;
%% Preprocess ECG signals 
% noise not present in the MIMIC-II dataset
% load('60Hz_norch_FIR_for_250.mat') % load in notch filter parameters for removing 60Hz power line interference noise
% ECG = filtfilt(Num,1,ECG); % notch filtering to remove 60Hz power line interference noise
%                            % 'Num' is the filter coefficients variable that
%                            % is loaded from '60Hz_norch_FIR_for_250.mat'
%% Detect ECG signal QRS peaks by method 1 and calculate heart rate (HR)
% Define parameters for QRS detection
% HR from ECG
% Time window for R peak correction.
% R peak detection
HRVparams.PeakDetect.REF_PERIOD = 0.25;    % Default: 0.25 (should be 0.15 for FECG), refractory period in sec between two R-peaks
HRVparams.PeakDetect.THRES = .6;           % Default: 0.6, Energy threshold of the detector
HRVparams.PeakDetect.fid_vec = [];         % Default: [], If some subsegments should not be used for finding the optimal
% threshold of the P&T then input the indices of the corresponding points here
HRVparams.PeakDetect.SIGN_FORCE = [];      % Default: [], Force sign of peaks (positive value/negative value)
HRVparams.PeakDetect.debug = 0;            % Default: 0
HRVparams.PeakDetect.ecgType = 'MECG';     % Default : MECG, options (adult MECG) or featl ECG (fECG)
HRVparams.PeakDetect.windows = 15;         % Befautl: 15,(in seconds) size of the window onto which to perform QRS detection
HRVparams.Fs = fsh;

% Detect QRS by run_qrsdet_by_seg() method
QRS = run_qrsdet_by_seg(ECG,HRVparams);

% Calculatae heart rates from detected QRSs
HR = 60./(diff(QRS)./fsh);

% Calculate median heart rate
medianHR = median(HR);

%% Preprocess Finapress signal (which is the golden standard of blood-pressure (BP) signals in our this measurement)
% Preprocess Finapress to remove 60 Hz power line interference noise
% m1 = mean(Finapres); % mean value of Finapress signal before filtering
% Finapres = filtfilt(Num,1,Finapres); % perform 60 Hz power line interference noise removing filtering
% m2 = mean(Finapres); % mean value of Finapress signal after filtering
% 
% Finapres = Finapres + m1 - m2; % adjust so that after filtering the signal would have similar mean value (baseline) as before filtering

% Preprocess Finapress signal, with the following major functionalities (see findsysdias0_qppg.m for details):
% a) Find Finapress diastolic blood pressure and systolic blood pressure envelopes, 
% b) Calculate Finapress Signal Quality Index (SQI),  c) detect beats down and
% up locations on the Finapress signals, as well as d) remove beats if SQI <
% sqi_th (75 here), and e) interpret to get back to original signal length
% after low quality beats removal and f) apply low pass fitering to the
% interpreted signal

% [Findbp0,Finsbp0]=findsysdias0(Finapres);
[Findbp,Finsbp,FinSQI,Finbeat_down,Finbeat_up] = findsysdias0_qppg(Finapres,fsh,medianHR,75);

% [Finsbp,Findbp]=nabpDiaSys(Finapres,fsh);
% [Finsbp,~]=nabpDiaSys(Finsbp,fsh);
% [~,Findbp]=nabpDiaSys(Findbp,fsh);


% Convert unit from A/D to mmHg
%Finapres = Finapres*0.2439-435.4; % TODO: check with team to see whether this applys to all subjects' Finapres signals
%Findbp = Findbp*0.2439-435.4; % TODO:as above
%Finsbp = Finsbp*0.2439-435.4; % TODO:as above
%MIMIC-II already in mmHg(Kevin)

FindbpSec = Findbp;
FinsbpSec = Finsbp;
FinmbpSec = (FinsbpSec+2*FindbpSec)/3;

%%%%
% Interpret Finapress SQI (FinSQI) to have the same length as that of the
% ECG, so that later on we will output this variable to the output .mat
% file, in that all variables will have the same length of ECG for later data analysis convenience.
% TODO: This part of code is very similar to that of the STAT SQI interpretation,
% to be refactored to implement inside a function, and call that
% function instead
finSQItVecSecond = ((1:length(Finapres))-1)/fsh; % convert to second unit

fintVecSecond4interp = [0; (Finbeat_up/fsh)'; finSQItVecSecond(end)]; 
FinSQIinterp4interp = [FinSQI(1);FinSQI;FinSQI(end)];

% "interp1" requires the values of "x" to be distinct, so get unique values of sqitVecSecond4interp
[~, ind3] = unique(fintVecSecond4interp);
fintVecSecond4interp = fintVecSecond4interp(ind3);
FinSQI4interp = FinSQIinterp4interp(ind3);
FinSQIinterp = interp1(fintVecSecond4interp, FinSQI4interp, finSQItVecSecond)';
%%%%

%% Perform ECG QRS detection by method 2 and Calculate ECG SQI based on method 1 and method2's QRS detection results
% SQI is Signal Quality Index

% Interpret HR to have the same length as that of the ECG signal
QRS(end) = []; % Remove the last element so that the below 'tHR' and 'HR' will have the same length for interp1()
tHR = [0 QRS length(ECG)];
HR = [HR(1) HR HR(end)];
HRinterp = interp1(tHR, HR, 1:length(ECG))'; % interpret HR to have the same length as that of the ECG signal
                                             % in sampling points, rather than seconds, in x axis

% QRS detection by wqrs (method 2)
signal2 = resample(ECG,125,fsh); % downsample to 125 Hz, as method 2 wqrsm_fast works the best on 125Hz signal
qrs_pos2 = wqrsm_fast(signal2,125); % detect QRS on the downsampled ECG signal
qrs_pos2 = round(qrs_pos2*(fsh/125)); % adjust back to QRS locations as ECG original samping rate of 250 Hz locations
qrs_pos2(find(qrs_pos2<1))=[];

% Calculate ECG's SQI by comparing the QRS detected by method 1 and method 2
% The idea is that if the signal quality is good, then the detected beats
% (QRS) would have similar locations by different detection methods;
% otherwise not with good quality

% bsqi
bsqi = zeros(1,floor(length(ECG)/fsh));
ann1 = QRS/fsh; % convert to second unit, as run_sqi_n.m requires it
ann2 = qrs_pos2/fsh; % convert to second unit, as run_sqi_n.m requires it
endtime = max([ann1(end) ann2(end)]);
% second-by-second bsqi, centered by the 10 sec window
for j = 0:round(endtime)-10
    [sqi] = run_sqi_n(ann1,ann2,0.1,j,10,fsh); % perform SQI calculation
    if j==0
        bsqi(1:j+5) = round(sqi*100);
    else
        bsqi(j+5) = round(sqi*100);
    end
end

bsqi(j+5:j+10) = round(sqi*100);
bsqi(find(isnan(bsqi))) = 0;
SQI = bsqi; % ECG SQI, x axis's unit is second
sqi_time = 1:length(SQI); % unit is second

% Interpret ECG's SQI to have same length of ECG (x axis, and in second
% unit in x axis, while y axis will the ECG's SQI values)
sqitVecSecond = ((1:length(ECG))-1)/fsh; % convert to second unit

sqitVecSecond4interp = [0;sqi_time';sqitVecSecond(end)]; % in second unit
SQI4interp = [SQI(1);SQI';SQI(end)];
% "interp1" requires the values of "x" to be distinct, so get unique values of sqitVecSecond4interp
[~, ind] = unique(sqitVecSecond4interp); % ind = index of first occurrence of a repeated value 
sqitVecSecond4interp = sqitVecSecond4interp(ind);
SQI4interp = SQI4interp(ind);
SQIinterp = interp1(sqitVecSecond4interp, SQI4interp, sqitVecSecond)';
ECGSQIinterp = SQIinterp; 

%% (MIMIC Contains no STAT)

%% NIRS preprocessing
% 
 [SQINIRSs, beats_NIRSs, nirsdata_all_channels, nirsdata_OD_all_channels, ...
           best_nirs_channel_data, best_nirs_OD_channel_data, best_nirs_channel_idx, best_nirs_channel_avgSQI] ...
                       = preprocessNIRS_calcNIRSSQI(NIRS, fsh);

 %plot beats_NIRSs 
 NIRS = nirsdata_all_channels;
 NIRS_OD = nirsdata_OD_all_channels;
 
 NIRS1 = best_nirs_channel_data;
 optData = best_nirs_OD_channel_data;
 NIRS1_OD = optData;
 
NIRS_best_channel_SQI = SQINIRSs{best_nirs_channel_idx}; % best NIRS channel SQI
NIRS_best_channel_beats = beats_NIRSs{best_nirs_channel_idx}; % best NIRS channel beats

% Interpret NIRS best channel SQI to have same length of ECG1 (x axis, and in second
% unit in x axis, while y axis will the NIRS best channel's SQI values)
% TODO: Refactor to put the below sqi interp part of code into a function,
% and call it instead; and similarly for ECG, STAT and Finapres' SQI
% interp
NIRS_best_channel_SQItVecSecond = ((1:length(ECG))-1)/fsh; % convert to second unit, to have the same length as that of the ECG,
                                            % which the same as that of the
                                            % NIRS too

NIRS_best_channel_tVecSecond4interp = [0; (NIRS_best_channel_beats/fsh)'; NIRS_best_channel_SQItVecSecond(end)]; 
NIRS_best_channel_SQIinterp4interp = [NIRS_best_channel_SQI(1);NIRS_best_channel_SQI;NIRS_best_channel_SQI(end)];

% "interp1" requires the values of "x" to be distinct, so get unique values of sqitVecSecond4interp
[~, ind2] = unique(NIRS_best_channel_tVecSecond4interp);
ind2(end) = [];
NIRS_best_channel_tVecSecond4interp = NIRS_best_channel_tVecSecond4interp(ind2);
NIRS_best_channel_SQI4interp = NIRS_best_channel_SQIinterp4interp(ind2);
NIRS_best_channel_SQIinterp = interp1(NIRS_best_channel_tVecSecond4interp, NIRS_best_channel_SQI4interp, NIRS_best_channel_SQItVecSecond)';


%There is only one channel of NIRS, so just applied OD calculation to PPG
%MIMIC II Data

%% Calculate PTT features by different algorithms (total of 4 algorithms here)
% Pulse transit time (PTT)

% Find the start calibration point value from Finapres
calBP_dbp = mean(FindbpSec(1:fsh*3));
calBP_sbp = mean(FinsbpSec(1:fsh*3));

tVecSec = ((1:length(ECG))-1)/fsh; % convert to in seconds unit

% Detect P peaks
[ppeakloc,optdata,channelNIRS,pir] = findPpeak2_update(optData,tVecSec,fsh,NIRS,1);

% Detect R peaks 
[rpeakloc, pairs, PTT,tVecptt] = findRpeakpairs(ppeakloc,ECG,fsh);

% Calculate PTT feature (BP calculated based on PTT and after calibration) by algorithm 1 and algorithm 2 respectively
[PTTsbp1,PTTdbp1,PTTsbp2,PTTdbp2] = PTT2BP_cal_1_2(fsh,PTT,pairs,calBP_sbp,calBP_dbp,NIRS1);
%[PTTsbp1,PTTdbp1,PTTsbp2,PTTdbp2]= PTT2BP_cal_1_2_ori(fsh,PTT,pairs,calBP_sbp,calBP_dbp,NIRS1);

% Calculate PTT feature by algorithm 3
[PTTsbp3,PTTdbp3] = PTT2BP_cal_3(fsh,PTT,pairs,calBP_sbp,calBP_dbp);

% Calculate PTT feature by algorithm 4
[PTTsbp4,PTTdbp4,~]= PTT2BP_cal(fsh,PTT,pairs,calBP_sbp,calBP_dbp);
% [PTTsbp,PTTdbp,~]= PTT2BP_cal_withPIR(fsh,PTT,pairs,calBP_sbp,calBP_dbp,pir);

% PTTMAD
tVecptt = [0;tVecptt;tVecSec(end)];
PTTsbp1 = [PTTsbp1(1);PTTsbp1;PTTsbp1(end)];
PTTsbpinterp1 = interp1(tVecptt, PTTsbp1, tVecSec)'; % interpret to have same length of ECG, so all features have the same length as of ECG
[bL, aL] = butter(3,1/(fsh/2)); % generate parameters for low pass filtering
PTTsbpinterp1 = filtfilt(bL,aL,PTTsbpinterp1); % low pass filtering

PTTdbp1 = [PTTdbp1(1);PTTdbp1;PTTdbp1(end)];
PTTdbpinterp1 = interp1(tVecptt,PTTdbp1,tVecSec)';
PTTdbpinterp1 = filtfilt(bL,aL,PTTdbpinterp1);

PTTmbpinterp1 = (2*PTTdbpinterp1+PTTsbpinterp1)/3;

PTTsbp2 = [PTTsbp2(1);PTTsbp2;PTTsbp2(end)];
PTTsbpinterp2 = interp1(tVecptt, PTTsbp2, tVecSec)';
PTTsbpinterp2 = filtfilt(bL,aL,PTTsbpinterp2);

PTTdbp2 = [PTTdbp2(1);PTTdbp2;PTTdbp2(end)];
PTTdbpinterp2 = interp1(tVecptt,PTTdbp2,tVecSec)';
PTTdbpinterp2 = filtfilt(bL,aL,PTTdbpinterp2);

PTTmbpinterp2 = (2*PTTdbpinterp2+PTTsbpinterp2)/3;

PTTsbp3 = [PTTsbp3(1);PTTsbp3;PTTsbp3(end)];
PTTsbpinterp3 = interp1(tVecptt, PTTsbp3, tVecSec)';
PTTsbpinterp3 = filtfilt(bL,aL,PTTsbpinterp3);

PTTdbp3 = [PTTdbp3(1);PTTdbp3;PTTdbp3(end)];
PTTdbpinterp3 = interp1(tVecptt,PTTdbp3,tVecSec)';
PTTdbpinterp3 = filtfilt(bL,aL,PTTdbpinterp3);

PTTmbpinterp3 = (2*PTTdbpinterp3+PTTsbpinterp3)/3;

PTTsbp4 = [PTTsbp4(1);PTTsbp4;PTTsbp4(end)];
PTTsbpinterp4 = interp1(tVecptt, PTTsbp4, tVecSec)';
PTTsbpinterp4 = filtfilt(bL,aL,PTTsbpinterp4);

PTTdbp4 = [PTTdbp4(1);PTTdbp4;PTTdbp4(end)];
PTTdbpinterp4 = interp1(tVecptt,PTTdbp4,tVecSec)';
PTTdbpinterp4 = filtfilt(bL,aL,PTTdbpinterp4);

PTTmbpinterp4 = (2*PTTdbpinterp4+PTTsbpinterp4)/3;

PTT = [PTT(1);PTT;PTT(end)];
PTTinterp = interp1(tVecptt,PTT,tVecSec)';

%No STAT Signal(Kevin)

% Adjust ECG's SQI to have the same seconds as that of the STAT
% signal. (STASQIinterp itself already has the same second lengths as that of the
% STAT signal, so no need to adjust it)
 l = length(ECG);
% nn = length(STAsbpcal(1:fsh:l)); % only need 1 SQI value per second
% if length(SQI)<nn
%     while length(SQI)<nn
%         SQI(end+1) = SQI(end);
%     end
% elseif length(SQI)>nn
%     while length(SQI)>nn
%         SQI(end)=[];
%     end
% end

%% Save preprocessed data to an output .mat file
% Get input .mat file's path, filename without extension information
[filePath, fileNameNoExt, ~] = fileparts(matFileFullPath);
disp("file name" + fileNameNoExt)
% Generate the output .mat file full path for preprocessed data saving
% Find the '_' in the string, as the filename may contain _unpreprocess, so
% we don't need the '_unpreprocess' in the output filename
% index = strfind(fileNameNoExt, '_');
% if (index>1) 
%     fileNameNoExt = fileNameNoExt(1:index-1);
% end

% Output data full path, the output directory will be the same as that of
% the input .mat file's directory 

sectionStartEndTimesStr = sprintf('_start%dsec_end%dsec', 1, 9999);

preprocessDataMatFileFullPath = fullfile(filePath, ['SOPData',fileNameNoExt, sectionStartEndTimesStr]); 
preprocessDataMatFileFullPath = "sampleFile_part1_start1sec";
% Prefix each to be saved variable's name with 'SOPData' for saving in the
% .mat output file
%ACCConv = DataACCConv; % as we will add "SOPData" to "ACCConv", instead of "SOPData" to "DataACCConv" that will result "SOPDataDataAccConv"
toBeSavedVariables_temp = {'ECG', 'ECGSQIinterp', 'HRinterp', ...
    'PTTsbpinterp1', 'PTTdbpinterp1', 'PTTsbpinterp2', 'PTTdbpinterp2', ...
    'PTTsbpinterp3', 'PTTdbpinterp3', 'PTTsbpinterp4', 'PTTdbpinterp4', 'PTTinterp', ...
    'Finapres','Finsbp','Findbp', 'FinSQIinterp', ...
    ...'STAT', 'STA','STAsbp','STAdbp', 'STASQIinterp', ...
    ...'STAcal', 'STAsbpcal', 'STAdbpcal', ... No STAT Calculations done
    'NIRS', 'NIRS_OD', 'optdata', 'pairs'...
    ...'NIRS1', 'NIRS1_OD', ... Only 1 NIRS Channel
    ...'ACCConv' %No Acc data
    };
toBeSavedVariables = strcat('SOPData', toBeSavedVariables_temp);

% Assign the to be saved variables to its corresponding variable names
% prefixed with SOPData, for example, SOPDataECG = ECG; This is for
% outputting to .mat files with 'SOPData' prefixed variables only convenience
for i = 1:length(toBeSavedVariables_temp)
   eval(sprintf('%s = %s;', toBeSavedVariables{i}, toBeSavedVariables_temp{i})); 
end

% Save different chanels' preprocessed data to the output .mat file
% Save 'SOPTimeVec' and 'SOPfsh' and all variables that are prefixed with 'SOPDATA' only
SOPTimeVec = tVecSec'; % time vector, convert to column vector, to consistent with other variables, 
                       % which are all column vectors, to be saved to the output file
SOPfsh = fsh; % sampling rate for saving in the output file
% save(preprocessDataMatFileFullPath, 'SOPTimeVec', 'SOPfsh', '-regexp','^SOPData');

% fprintf('\nSaved preprocessed data to: %s\n', preprocessDataMatFileFullPath);

%% Concatenate different features and output to a feature output .mat file
Xdata = [
    ...STAsbpcal(1:fsh:l)';STAdbpcal(1:fsh:l)';STASQIinterp(1:fsh:l)';
    HRinterp(1:fsh:l)'; ...
    PTTsbpinterp1(1:fsh:l)';PTTdbpinterp1(1:fsh:l)'; ...
    PTTsbpinterp2(1:fsh:l)';PTTdbpinterp2(1:fsh:l)'; ...
    PTTsbpinterp3(1:fsh:l)';PTTdbpinterp3(1:fsh:l)'; ...
    PTTsbpinterp4(1:fsh:l)';PTTdbpinterp4(1:fsh:l)'; ...
    PTTinterp(1:fsh:l)'; 
    %Adding NIRS, ECG, and SQI data for NIRS ECG and Finapres(Kevin 11/27)
    best_nirs_OD_channel_data(1:fsh:l)'; ECG(1:fsh:l)';... 
    NIRS_best_channel_SQIinterp(1:fsh:l)'; ECGSQIinterp(1:fsh:l)';...
     FinSQIinterp(1:fsh:l)'; ...
    %Adding NIRS and finapres SQI
    ...SQINIRSs; FinSQI; ...
    
    
    ...STAsbp(1:fsh:l)';STAdbp(1:fsh:l)'
    ];


%Xdata(3,:) = medfilt1(Xdata(3,:),9);
%Xdata(4,:) = medfilt1(Xdata(4,:),9);
Ydata = [FinsbpSec(1:fsh:l)';FindbpSec(1:fsh:l)'];

Xdata = Xdata';
Ydata = Ydata';