function P = constructCovMat(X,regularization)
% CONSTRUCTCOVMAT Compute a covariance matrix of a given EEG data matrix 
% regularization method.
%
%   Input parameters:
%       X [DOUBLE]: EEG tensor (channel x time x trial)
%       regularization [STRUCT]: struct with fields
%           - type [STRING]: 'none' (no regularization) or 'ridge' (ridge
%           regression)
%           - lambda [DOUBLE]: ridge regression hyperparameter

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

P = zeros(size(X,1),size(X,1),size(X,3));
if strcmp(regularization.type,'none')
    for tr = 1:size(X,3)
        P(:,:,tr) = cov(X(:,:,tr)');
        P(:,:,tr) = (P(:,:,tr)+P(:,:,tr)')/2;
    end
elseif strcmp(regularization.type,'ridge')
    for tr = 1:size(X,3)
        Rxx = X(:,:,tr)*X(:,:,tr)';
        P(:,:,tr) = Rxx+regularization.lambda*trace(Rxx)/(size(X(:,:,tr),1))*eye(size(X(:,:,tr),1));
        P(:,:,tr) = (P(:,:,tr)+P(:,:,tr)')/2;
    end
else
    error('Invalid regularization type');
end
    
end