function [predLabel,riemDist] = mdrm(P,labels,params)
% MDRM Classify according to minimum distance to Riemannian mean.
%
%   Input parameters:
%       P [STRUCT]:
%           train [DOUBLE]: tensor of training covariance matrices 
%           (channel x channel x trial)
%           test [DOUBLE]: tensor of test covariance matrices 
%           (channel x channel x trial)
%       labels [STRUCT]:
%           train [INTEGER]: labels corresponding to the attended location
%           (1 or 2) of the training covariance matrices
%       params [STRUCT]:
%               method [STRING]: 'log-euclidean' (approximation) or
%               'riemannian' (affine-invariant)
%               epsilon [DOUBLE]: stopping criterion parameter for 
%               log-euclidean method

% Authors: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

%% compute Riemannian mean per class
Pm = zeros(size(P.train,1),size(P.train,2),length(unique(labels.train)));
for class = 1:size(Pm,3)
    Pm(:,:,class) = computeRiemannianMean(P.train(:,:,labels.train==class),params);
end

%% classification: minimal Riemannian distance
predLabel = zeros(size(P.test,3),1);
riemDist = zeros(size(P.test,3),2);
for tr = 1:size(P.test,3)
    dist = zeros(size(Pm,3),1);
    for class = 1:size(Pm,3)
        dist(class) = riemannianDist(P.test(:,:,tr),Pm(:,:,class));
    end
    [~,predLabel(tr)] = min(dist);
    riemDist(tr,:) = dist;
end

end