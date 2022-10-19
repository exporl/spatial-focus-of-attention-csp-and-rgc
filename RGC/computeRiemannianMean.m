function Pm = computeRiemannianMean(P,params)
% COMPUTERIEMANNIANMEAN Compute Riemannian mean of set of covariance
% matrices.
%
%   Input parameters:
%       P [DOUBLE]: tensor of covariance matrices (channel x channel x trial)
%       params [STRUCT]:
%               method [STRING]: 'log-euclidean' (approximation) or
%               'riemannian' (affine-invariant)
%               epsilon [DOUBLE]: stopping criterion parameter for 
%               log-euclidean method

% Authors: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

%% log-euclidean method (approximation)
if strcmp(params.method,'log-euclidean')
    for tr = 1:size(P,3)
       [U,D] = eig(P(:,:,tr)); 
       P(:,:,tr) = U*diag(log(diag(D)))*U';
    end
    Pm = mean(P,3);
    [U,D] = eig(Pm);
    Pm = U*diag(exp(diag(D)))*U';
%% riemannian method (exact)
elseif strcmp(params.method,'riemannian')
    Pm = mean(P,3);
    S = Inf*ones(size(P,1),size(P,2));
    while norm(S,'fro') > params.epsilon
        Pmsqmin = mpower(Pm,-1/2); Pmsqmin = (Pmsqmin+Pmsqmin')/2;
        logInner = tmprod(tmprod(P,Pmsqmin,1),Pmsqmin,2);
        for tr = 1:size(logInner,3)
            logInner(:,:,tr) = (logInner(:,:,tr)+logInner(:,:,tr)')/2;
            [U,D] = eig(logInner(:,:,tr));
            logInner(:,:,tr) = U*diag(log(diag(D)))*U';
        end
        Pmsq = mpower(Pm,1/2); Pmsq = (Pmsq+Pmsq')/2;
        logMap = tmprod(tmprod(logInner,Pmsq,1),Pmsq,2); 
        S = mean(logMap,3);
        Pm = Pmsq*expm(Pmsqmin*S*Pmsqmin)*Pmsq; Pm = (Pm+Pm')/2;
    end
else
    error('Invalid Riemannian mean computation method');
end
end