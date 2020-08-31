function preprocessData(basedir)
% PREPROCESS_DATA Preprocesses EEG data as described in the paper:
%   Fast EEG-based decoding of the directional focus of auditory attention using common spatial patterns
%   Simon Geirnaert, Tom Francart, Alexander Bertrand, bioRxiv 2020.06.16.154450; doi: https://doi.org/10.1101/2020.06.16.154450
%
%   Input parameters:
%       basedir [STRING]: the directory in which all the subject and stimuli data
%                         are saved. (Default: current folder)
%   Dependency:
%       Tensorlab (https://www.tensorlab.net/)

% Authors: Neetha Das/Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

if nargin == 0
    basedir = pwd;
end

stimulusdir = [basedir filesep 'stimuli'];
envelopedir = [stimulusdir filesep 'envelopes'];
if ~exist(envelopedir,'dir')
    mkdir(envelopedir);
end

% Set parameters
params.intermediateSampleRate = 128; %Hz
params.lowpass = 32; % Hz, used for constructing a bpfilter used for both the audio and the eeg
params.highpass = 1; % Hz
params.targetSampleRate = 64; % Hz
params.rereference = 'none';
params.segSize = 60; % 60 second segments

% Build the bandpass filter
bpFilter = construct_bpfilter(params);

%% Preprocess EEG and put EEG and corresponding stimulus envelopes together

preprocdir = [basedir filesep 'preprocessed_data'];
if ~exist(preprocdir,'dir')
    mkdir(preprocdir)
end
subjects = dir([basedir filesep 'S*.mat']);
subjects = sort({subjects(:).name});

for subject = subjects
    load(fullfile(basedir,subject{1}))
    eegTrials = {};
    conditions = [];
    attendedEar = [];
    
    for trialnum = 1:size(trials,2)
        
        trial = trials{trialnum};
        
        % Rereference the EEG data if necessary
        if strcmpi(params.rereference,'Cz')
            trial.RawData.EegData = trial.RawData.EegData - repmat(trial.RawData.EegData(:,48),[1,64]);
        elseif strcmpi(params.rereference,'mean')
            trial.RawData.EegData = trial.RawData.EegData - repmat(mean(trial.RawData.EegData,2),[1,64]);
        end
        
        % Apply the bandpass filter
        trial.RawData.EegData = filtfilt(bpFilter.numerator,1,double(trial.RawData.EegData));
        trial.RawData.HighPass = params.highpass;
        trial.RawData.LowPass = params.lowpass;
        trial.RawData.bpFilter = bpFilter;
        
        % downsample EEG (using downsample so no filtering appears).
        downsamplefactor = trial.FileHeader.SampleRate/params.targetSampleRate;
        if round(downsamplefactor)~= downsamplefactor, error('Downsamplefactor is not integer'); end
        trial.RawData.EegData = downsample(trial.RawData.EegData,downsamplefactor);
        trial.FileHeader.SampleRate = params.targetSampleRate;
        
        % split into segments
        eegSegments = segmentize(trial.RawData.EegData,'SegSize',params.segSize*params.targetSampleRate);
        for str = 1:size(eegSegments,2)
            eegTrials = [eegTrials;{squeeze(eegSegments(:,str,:))}];
            if strcmp(trial.attended_ear,'L')
                attendedEar = [attendedEar;1];
            else
                attendedEar = [attendedEar;2];
            end
        end
        fs = params.targetSampleRate;
    end
    save(fullfile(preprocdir,['data',subject{1}]),'eegTrials','attendedEar','fs')
end

end

function [ BP_equirip ] = construct_bpfilter( params )

Fs = params.intermediateSampleRate;
Fst1 = params.highpass-0.45;
Fp1 = params.highpass+0.45;
Fp2 = params.lowpass-0.45;
Fst2 = params.lowpass+0.45;
Ast1 = 20; %attenuation in dB
Ap = 0.5;
Ast2 = 15;
BP = fdesign.bandpass('Fst1,Fp1,Fp2,Fst2,Ast1,Ap,Ast2',Fst1,Fp1,Fp2,Fst2,Ast1,Ap,Ast2,Fs);
BP_equirip = design(BP,'equiripple');

end




