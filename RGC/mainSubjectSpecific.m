%% MAIN SUBJECT-SPECIFIC RIEMANNIAN CLASSIFIER
% Main script for subject-specific RGC algorithm, as described in:
%   S. Geirnaert, T. Francart and A. Bertrand, "Riemannian Geometry-Based 
%       Decoding of the Directional Focus of Auditory Attention Using EEG,"
%       ICASSP 2021 - 2021 IEEE International Conference on Acoustics, 
%       Speech and Signal Processing (ICASSP), 2021, pp. 1115-1119, 
%       doi: 10.1109/ICASSP39728.2021.9413404.

%
% Dependency:
%       Tensorlab (https://www.tensorlab.net/)
%
% Authors: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

clear; close all;


%% Setup: parameters
params.dataset = 'das-2016'; % 'das-2016' (16 subj) / 'fuglsang2018' (18 subj) / 'rob' (12 subj) / '256' (30 subj) / 'fuglsang2020' (44 subj) / 'aad-in-noise' (18 sujects) / 'switch-data-2020' (3 subjects)
params.subjects = 1:16; % subjects to test
params.windowLengths = [60,30,20,10,5,2,1]; % different lengths decision windows to test (in seconds)
params.save = true; % save results or not
params.saveName = '64ch-beta'; % name to save results with

% preprocessing
params.preprocessing.normalization = false; % 1: with normalization of regression matrices (column-wise), 0: without normalization
params.preprocessing.rereference = 'none'; % 'none' / 'Cz' / 'CAR' / 'custom'
params.preprocessing.eegChanSel = []; % array of channels to select

% bandpass filter
params.filterbands = [12;30]; % beta band

% covariance construction
params.cov.method = 'lwcov'; % covariance matrix estimation method: 'cov' / 'lwcov'

% riemannian mean parameters
params.riem.method = 'log-euclidean'; % 'riemannian' (affine-invariant) / 'log-euclidean' (approximation)
params.riem.epsilon = 1e-12; % stopping criterion parameter for log-euclidean method

% cross-validation
params.cv.nfold = 10; % number of random folds in every CV repetition
params.cv.nrep = 1; % repetitions of CV procedure

% classification parameters (only valid for 'tsm')
params.riem.class.method = 'tsm'; % 'mdrm': minimal distance to Riemannian mean / 'tsm': tangent space mapping + classification
params.class.method = 'svm'; % 'lda' / 'svm'
params.class.kernel = 'linear';
params.class.optimized = false; % optimization hyperparameters

%% Setup: parameter processing

% optimization classifier training
if strcmp(params.class.method,'svm')
    arg = {'Prior','uniform'}; % prior
    arg = [arg,{'KernelFunction',params.class.kernel}]; % kernel function
    arg = [arg,{'Verbose',0}];
    arg = [arg,{'Standardize',false}]; % standardization
elseif strcmp(params.class.method,'lda')
    arg = {'Prior','uniform'}; % prior
end
    
    
% optimization classifier training
if params.class.optimized
    arg = [arg,{'OptimizeHyperparameters','auto'}]; % optimization of hyperparameters
    optOptions = struct;
    optOptions.Kfold = 5;
    optOptions.MaxObjectiveEvaluations = 20;
    optOptions.Verbose = 0;
    optOptions.ShowPlots = true;
    params.class.arg = [arg,{'HyperparameterOptimizationOptions',optOptions}];
else
    params.class.arg = arg;
end

% construct a results variable
results = struct;
results.testacc = zeros(params.cv.nrep,params.cv.nfold,length(params.subjects),length(params.windowLengths));
results.trainacc = zeros(params.cv.nrep,params.cv.nfold,length(params.subjects),length(params.windowLengths));

%% Loop over subjects
for testSubj = 1:length(params.subjects)
    fprintf('\n%s\n*** Testing subject %d ***\n%s\n',repmat('-',1,30),params.subjects(testSubj),repmat('-',1,30))
        
    % load data of test subject
    testS = params.subjects(testSubj);
    [eeg,attendedEar,fs,trialLength] = loadData(params.dataset,testS,params.preprocessing);

    % apply filtering
    d = designfilt('bandpassiir','FilterOrder',8, ...
        'HalfPowerFrequency1',params.filterbands(1),'HalfPowerFrequency2',params.filterbands(2), ...
        'SampleRate',fs);
    eeg = permute(filtfilt(d,permute(eeg,[2,1,3])),[2,1,3]);

    % cross-validation
    for rep = 1:params.cv.nrep
        fprintf('\n%s\n*** Repetition nr. %d ***\n%s\n',repmat('-',1,30),rep,repmat('-',1,30))
        
        % generate a division of the data in folds
        c{rep} = cvpartition(attendedEar,'Kfold',params.cv.nfold);
        
        % loop over CV folds
        for fold = 1:params.cv.nfold
            fprintf('\n%s\n fold nr. %d\n%s\n',repmat('-',1,15),fold,repmat('-',1,15))
            
            % generate a split in (training+validation)/testing data
            idx.train = c{rep}.training(fold);
            idx.test = c{rep}.test(fold);
            
            X = struct;
            X.test = eeg(:,:,idx.test);
            X.train = eeg(:,:,idx.train);
            
            labels = struct;
            labels.test = attendedEar(idx.test);
            labels.train = attendedEar(idx.train);
            
            for w = 1:length(params.windowLengths)
                
                %% segment data into windows of given length
                Xw.train = permute(X.train,[2,1,3]);
                Xw.train = segmentize(Xw.train,'Segsize',params.windowLengths(w)*fs);
                Xw.train = permute(Xw.train,[3,1,2,4]);
                Xw.train = reshape(Xw.train,[size(Xw.train,1),size(Xw.train,2),size(Xw.train,3)*size(Xw.train,4)]);
                labelsw.train = repelem(labels.train,floor(trialLength/(params.windowLengths(w)*fs)));
                labelsw.train = labelsw.train(:);
                
                Xw.test = permute(X.test,[2,1,3]);
                Xw.test = segmentize(Xw.test,'Segsize',params.windowLengths(w)*fs);
                Xw.test = permute(Xw.test,[3,1,2,4]);
                Xw.test = reshape(Xw.test,[size(Xw.test,1),size(Xw.test,2),size(Xw.test,3)*size(Xw.test,4)]);
                labelsw.test = repelem(labels.test,floor(trialLength/(params.windowLengths(w)*fs)));
                labelsw.test = labelsw.test(:);
                
                %% construct covariance matrices
                if strcmp(params.cov.method,'cov')
                    P.train = constructCovMat(Xw.train,params.cov.regularization);
                    P.test = constructCovMat(Xw.test,params.cov.regularization);
                elseif strcmp(params.cov.method,'lwcov')
                    P.train = zeros(size(Xw.train,1),size(Xw.train,1),size(Xw.train,3));
                    for tr = 1:size(Xw.train,3)
                        P.train(:,:,tr) = lwcov(Xw.train(:,:,tr)');
                        P.train(:,:,tr) = (P.train(:,:,tr)+P.train(:,:,tr)')/2;
                    end
                    P.test = zeros(size(Xw.test,1),size(Xw.test,1),size(Xw.test,3));
                    for tr = 1:size(Xw.test,3)
                        P.test(:,:,tr) = lwcov(Xw.test(:,:,tr)');
                        P.test(:,:,tr) = (P.test(:,:,tr)+P.test(:,:,tr)')/2;
                    end
                else
                    error('Invalid covariance matrix estimator');
                end             
                
                %% classify according to chosen method
                if strcmp(params.riem.class.method,'mdrm')
                    [predicted.test,riemDist] = mdrm(P,labelsw,params);
                    results.testacc(rep,fold,testSubj,w) = mean(labelsw.test == predicted.test);
                elseif strcmp(params.riem.class.method,'tsm')
                    % compute Riemannian mean
                    Pm = computeRiemannianMean(P.train,params.riem);
                    Pmsqmin = mpower(Pm,-1/2);
                    hvMat = ones(size(P.train,1));
                    hvMat(triu(true(size(hvMat)),1)) = sqrt(2);
                    
                    % compute features
                    n = size(P.train,1);
                    f.train = zeros(size(P.train,3),n*(n+1)/2);
                    R.train = zeros(n,n,size(P.train,3));
                    logInnerTrain = tmprod(tmprod(P.train,Pmsqmin,1),Pmsqmin,2);
                    
                    for tr = 1:size(logInnerTrain,3)
                        logMap = logm(logInnerTrain(:,:,tr));
                        
                        % take upper triangular part as feature vector
                        R.train(:,:,tr) = logMap;
                        logMap = logMap.*hvMat;
                        f.train(tr,:) = logMap(triu(true(size(logMap))));
                    end
                    
                    f.test = zeros(size(P.test,3),n*(n+1)/2);
                    R.test = zeros(n,n,size(P.test,3));
                    logInnerTest = tmprod(tmprod(P.test,Pmsqmin,1),Pmsqmin,2);
                    for tr = 1:size(logInnerTest,3)
                        logMap = logm(logInnerTest(:,:,tr));
                        
                        % take upper triangular part as feature vector
                        R.test(:,:,tr) = logMap;
                        logMap = logMap.*hvMat;
                        f.test(tr,:) = logMap(triu(true(size(logMap))));
                    end
                    
                    % classifier training
                    if strcmp(params.class.method,'svm')
                        model = fitcsvm(f.train,labelsw.train,params.class.arg{:});
                    elseif strcmp(params.class.method,'lda')
                        model = fitcdiscr(f.train,labelsw.train,params.class.arg{:});
                    else
                        error('Choose valid classifier')
                    end
                    
                    % prediction
                    predicted.train = predict(model,f.train);
                    predicted.test = predict(model,f.test);
                    
                    results.trainacc(rep,fold,testSubj,w) = mean(labelsw.train == predicted.train);
                    results.testacc(rep,fold,testSubj,w) = mean(labelsw.test == predicted.test);
                end
            end
        end
        if params.save
            save(['results-',params.dataset,'-',params.saveName],'results');
        end
        disp(squeeze(mean(mean(results.testacc,1),2)))
    end
end

%% Results aggregation
acc_test = squeeze(mean(mean(results.testacc,1),2))
acc_train = squeeze(mean(mean(results.trainacc,1),2))


results.params = params;
if params.save
    save(['results-',params.dataset,'-',params.saveName],'results','acc_train','acc_test','params');
end