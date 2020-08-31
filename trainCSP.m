function [W,score,tr] = trainCSP(X,label,npat,optmode,heuristicPatSel,covMethod)
% TRAINCSP Estimate a number of Common Spatial(-Spectral) Pattern filters
% to filter EEG data in a supervised fashion, for feature extraction.
%
%   Note: 2 ways of estimating the CSP filters are provided.
%   1) based on GEVD (classical CSP, as in BCI literature)
%   2) based on trace ratio optimization (cfr. [1])
%
%   Input parameters:
%       X [DOUBLE]: EEG tensor (channel x time x trial)
%                   label [1 or 2]: trial labels (binary)
%       npat [INTEGER]: number of CS patterns to retain (in total)
%       optmode [STRING]: 'ratiotrace' (standard CSP) or 'traceratio'
%                         (traceratio optimization)
%       heuristicPatSel [LOGICAL]: BBCI heuristic to select filters or not
%       covMethod [STRING]: 'cov' (classical sample covariance) or 'lwcov'
%                           (ledoit-wolf estimation)
%
%   Dependency:
%       Tensorlab (https://www.tensorlab.net/)
%
%   References:
%   Ngo, Thanh T., Mohammed Bellalij, and Yousef Saad.
%   "The trace ratio optimization problem." SIAM review 54.3 (2012): 545-569.

% Authors: Simon Van Eyndhoven, KU Leuven, ESAT
%          Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

%% Initialization

% default parameters
if (nargin < 4) || ~any(strcmp(optmode,{'traceratio','ratiotrace'}))
    optmode = 'traceratio';
    covMethod = 'cov';
    heuristicPatSel = true;
else
end

% selection of filter outputs
if nargin < 3
    patidx = 1:size(X,1);
else
    patidx = [1:ceil(npat/2),size(X,1)-ceil(npat/2)+1:size(X,1)];
end

% assert only two classes
yc = unique(label);
assert(numel(yc)==2) % CSP defined for 2 class-problem

%% Split the data according to the training class labels
X1 = X(:,:,label==yc(1));
X2 = X(:,:,label==yc(2));
assert(size(X1,3)+size(X2,3)==size(X,3))

%% Compute class covariances
Xm1 = tens2mat(X1,1,[]);
Xm2 = tens2mat(X2,1,[]);

switch covMethod
    case 'lwcov'
        S1 = lwcov(Xm1');
        S2 = lwcov(Xm2');
    case 'cov'
        S1 = cov(Xm1');
        S2 = cov(Xm2');
end

% normalization
S1 = S1/trace(S1);
S2 = S2/trace(S2);

%% Optimize CSP filters
if strcmp(optmode,'ratiotrace') % 'standard CSP'
    
    % compute CSP filters using GEVD
    [W,D] = eig(S1,S1+S2);
    lambda = diag(D);
    
    % BBCI heuristic
    if heuristicPatSel
        Y1 = tmprod(X1,W',1);
        Y2 = tmprod(X2,W',1);
        Y1 = squeeze(var(Y1,[],2));
        Y2 = squeeze(var(Y2,[],2));
        score = median(Y1,2)./(median(Y1,2) + median(Y2,2));
    else
        score = lambda;
    end
    [score,order] = sort(score,'descend');
    W = W(:,order);
    
    % truncate to the desired number of CSP filters
    W = W(:,patidx);
    score = score(patidx);
    
    tr = zeros(1,2);
    tr(1) = trace(W'*S1*W)/trace(W'*(S1+S2)*W);
    tr(2) = trace(W'*S2*W)/trace(W'*(S1+S2)*W);
    
elseif strcmp(optmode,'traceratio') % 'optimized CSP'
    
    % initialize
    npathalf = round(npat/2);
    
    % compute CSP filters for class 1
    W1 = randn(size(S1,1),npathalf);
    [W1,~] = qr(W1);
    
    relchange = Inf;
    tr0 = Inf;
    while relchange > 1e-3
        tr = trace(W1'*S1*W1)/trace(W1'*(S1+S2)*W1);
        [temp,D] = eig(S1-tr*(S1+S2));
        lambda = diag(D);
        [lambda,order] = sort(lambda,'descend');
        temp = temp(:,order);
        lambda = lambda(1:npathalf);
        W1 = temp(:,1:npathalf);
        
        relchange = abs(tr-tr0)/tr;
        tr0 = tr;
    end
    
    score = lambda;
    
    % compute CSP filters for class 2
    W2 = randn(size(S1,1),npathalf);
    [W2,~] = qr(W2);
    
    relchange = Inf;
    tr0 = Inf;
    while relchange > 1e-3
        tr = trace(W2'*S2*W2)/trace(W2'*(S1+S2)*W2);
        [temp,D] = eig(S2-tr*(S1+S2));
        lambda = diag(D);
        [lambda,order] = sort(lambda,'descend');
        temp = temp(:,order);
        lambda = lambda(1:npathalf);
        W2 = temp(:,1:npathalf);
        
        relchange = abs(tr-tr0)/tr;
        tr0 = tr;
    end
    
    score = [score;lambda];
    
    W = [W1,W2];
    
    tr = zeros(1,2);
    tr(1) = trace(W1'*S1*W1)/trace(W1'*(S1+S2)*W1);
    tr(2) = trace(W2'*S2*W2)/trace(W2'*(S1+S2)*W2);
    
else
    error('Invalid CSP optimization mode')
end

end