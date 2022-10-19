%% MAIN SUBJECT-SPECIFIC FILTERBANK CSP FILTER
% Main script for subject-specific FB-CSP filtering, as described in:
%   S. Geirnaert, T. Francart and A. Bertrand, "Fast EEG-Based Decoding Of 
%       The Directional Focus Of Auditory Attention Using Common Spatial 
%       Patterns," in IEEE Transactions on Biomedical Engineering, vol. 68,
%       no. 5, pp. 1557-1568, May 2021, doi: 10.1109/TBME.2020.3033446.

%
% Dependency:
%       Tensorlab (https://www.tensorlab.net/)
%
% Authors: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
%          Simon Van Eyndhoven, KU Leuven, ESAT
% Correspondence: simon.geirnaert@esat.kuleuven.be

clear; close all;

%% Setup: parameters
params.dataset = 'das-2016'; % 'das-2016' (16 subjects) / ... (other datasets to be completed)
params.subjects = 1:16; % subjects to test
params.windowLengths = [60,30,20,10,5,2,1]; % different lengths decision windows to test (in seconds)
params.save = true; % save results or not
params.saveName = 'FB-64ch-beta'; % name to save results with

% preprocessing
params.preprocessing.normalization = true; % 1: normalization of EEG trials (Frobenius norm + centering), 0: without normalization
params.preprocessing.rereference = 'none'; % 'none' / 'Cz' / 'CAR' / 'custom'
params.preprocessing.eegChanSel = []; % array of channels to select

% filterbank setup
% params.filterbank.bands = [1,2:2:26;4:2:30]; % first row: lower bound, second row: upper bound
params.filterbank.bands = [12;30]; % beta band

% covariance estimation
params.cov.method = 'lwcov'; % covariance matrix estimation method: 'cov' / 'lwcov'

% CSP filters
params.csp.npat = 6; % number of CS patterns to retain (in total, per band)
params.csp.optmode = 'ratiotrace'; % optimization mode: 'ratiotrace' or 'traceratio'
params.csp.heuristicPatSel = true; % BBCI heuristic to select CSP filters

% cross-validation
params.cv.nfold = 10; % number of random folds in every CV repetition
params.cv.nrep = 1; % repetitions of CV procedure

% classification parameters
params.class.method = 'lda'; % 'lda' / 'svm'
params.class.optimized = false; % optimization hyperparameters

%% Setup: parameter processing

% optimization classifier training
if params.class.optimized
    arg = {'OptimizeHyperparameters','auto'}; % optimization of hyperparameters
    arg = [arg,{'Prior','uniform'}]; % prior
    optOptions = struct;
    optOptions.Kfold = 5;
    optOptions.MaxObjectiveEvaluations = 20;
    optOptions.Verbose = 0;
    optOptions.ShowPlots = false;
    params.class.arg = [arg,{'HyperparameterOptimizationOptions',optOptions}];
    if strcmp(params.class.method,'svm')
        arg = [arg,{'KernelFunction','linear'}]; % kernel function
        arg = [arg,{'Verbose',0}];
        arg = [arg,{'Standardize',false}]; % standardization
    end
else
    params.class.arg = {'Prior','uniform'};
end

% construct a results variable
results = struct;
results.testacc = zeros(params.cv.nrep,params.cv.nfold,length(params.subjects),length(params.windowLengths));
results.trainacc = zeros(params.cv.nrep,params.cv.nfold,length(params.subjects),length(params.windowLengths));

%% Loop over test subjects
for testSubj = 1:length(params.subjects)
    fprintf('\n%s\n*** Testing subject %d ***\n%s\n',repmat('-',1,30),params.subjects(testSubj),repmat('-',1,30))
    
    % load data of test subject
    testS = params.subjects(testSubj);
    [eeg,attendedEar,fs,trialLength] = loadData(params.dataset,testS,params.preprocessing);
    
    % apply filterbank
    eegTemp = eeg;
    eeg = zeros(size(eeg,1),size(params.filterbank.bands,2),size(eeg,2),size(eeg,3)); % channel x band x time x trial
    for band = 1:size(params.filterbank.bands,2)
        d = designfilt('bandpassiir','FilterOrder',8, ...
            'HalfPowerFrequency1',params.filterbank.bands(1,band),'HalfPowerFrequency2',params.filterbank.bands(2,band), ...
            'SampleRate',fs);
        eeg(:,band,:,:) = permute(filtfilt(d,permute(eegTemp,[2,1,3])),[2,1,3]);
    end
    clear('eegTemp');
    
    % cross-validation
    for rep = 1:params.cv.nrep
        fprintf('\n%s\n*** Repetition nr. %d ***\n%s\n',repmat('-',1,30),rep,repmat('-',1,30))
        
        % generate a division of the data in folds
        c{rep} = cvpartition(attendedEar,'Kfold',params.cv.nfold);
        
        % loop over CV folds
        for fold = 1:params.cv.nfold
            fprintf('\n%s\n fold nr. %d\n%s\n',repmat('-',1,15),fold,repmat('-',1,15))
            
            % generate a split in training/testing data
            idx.train = c{rep}.training(fold);
            idx.test = c{rep}.test(fold);
            
            X = struct;
            X.test = eeg(:,:,:,idx.test);
            X.train = eeg(:,:,:,idx.train);
            
            labels = struct;
            labels.test = attendedEar(idx.test);
            labels.train = attendedEar(idx.train);
            
            %% Train CSP filters
            CSP = struct;
            CSP.W = []; CSP.score = []; CSP.traceratio = [];
            Y = struct; Y.train = []; Y.test = [];
            for band = 1:size(params.filterbank.bands,2)
                % train CSP
                [W,score,traceratio] = trainCSP(squeeze(X.train(:,band,:,:)),labels.train,params.csp.npat,params.csp.optmode,params.csp.heuristicPatSel,params.cov.method);
                CSP.W = cat(3,CSP.W,W);
                CSP.score = cat(3,CSP.score,score);
                CSP.traceratio = cat(3,CSP.traceratio,traceratio);
                
                % filter both training and testing data using the CSP filters
                Y.train = cat(4,Y.train,tmprod(squeeze(X.train(:,band,:,:)),CSP.W(:,:,band)',1));
                Y.test = cat(4,Y.test,tmprod(squeeze(X.test(:,band,:,:)),CSP.W(:,:,band)',1));
            end
            
            Y.train = permute(Y.train,[1,4,2,3]);
            Y.test = permute(Y.test,[1,4,2,3]);
            
            %% Train and test classifier
            for w = 1:length(params.windowLengths)
                % segment data into windows of given length
                Y.windowed.train = segment(permute(Y.train,[3,1,2,4]),params.windowLengths(w)*fs);
                labels.windowed.train = repelem(labels.train,floor(trialLength/(params.windowLengths(w)*fs)),1);
                Y.windowed.test = segment(permute(Y.test,[3,1,2,4]),params.windowLengths(w)*fs);
                labels.windowed.test = repelem(labels.test,floor(trialLength/(params.windowLengths(w)*fs)),1);
                
                % feature extraction: log(energy) of CSP filter outputs
                feat = struct;
                feat.train = log(sum(Y.windowed.train.^2,3));
                feat.train = reshape(feat.train,[size(feat.train,1)*size(feat.train,2),size(feat.train,4)])';
                feat.test = log(sum(Y.windowed.test.^2,3));
                feat.test = reshape(feat.test,[size(feat.test,1)*size(feat.test,2),size(feat.test,4)])';
                
                % classifier training
                if strcmp(params.class.method,'svm')
                    model = fitcsvm(feat.train,labels.windowed.train,params.class.arg{:});
                elseif strcmp(params.class.method,'lda')
                    model = fitcdiscr(feat.train,labels.windowed.train,params.class.arg{:});
                else
                    error('Choose valid classifier')
                end
                
                % prediction
                predicted.train = predict(model,feat.train);
                predicted.test = predict(model,feat.test);
                
                results.trainacc(rep,fold,testSubj,w) = mean(labels.windowed.train == predicted.train);
                results.testacc(rep,fold,testSubj,w) = mean(labels.windowed.test == predicted.test);
            end
        end
        
        % save intermediate results
        if params.save
            save(['results-',params.dataset,'-',params.saveName],'results');
        end
        disp(squeeze(mean(mean(results.testacc,1),2)))
    end
end

%% Results aggregation
acc_train = squeeze(mean(mean(results.trainacc,1),2))
acc_test = squeeze(mean(mean(results.testacc,1),2))

results.params = params;
if params.save
    save(['results-',params.dataset,'-',params.saveName],'results','acc_train','acc_test','params');
end