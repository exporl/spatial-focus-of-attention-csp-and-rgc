function Rxx = lwcov(X)
% LWCOV Compute a well-conditioned (regularized) covariance matrix
% according to the method by Ledoit & Wolf [1], i.e., by computing a new
% covariance matrix as a weighted combination of the (poor-conditioned, but
% unbiased) sample covariance matrix and the (well-conditioned, but
% uninformative) identity matrix (= shrinkage).
% - The regularized covariance is the optimal linear shrinkage in the
% minimum mean squared error sense: it minimizes the expected squared
% error deviation from the true but unknown covariance, and as such is at
% the optimal point of the 'bias-variance' tradeoff that is omnipresent in
% statistics and machine learning (cfr. eq. (10) in [1]).
% - Since a linear combination with an identity matrix is made, the
% eigenvectors (~ principal subspaces) of the covariance are not altered,
% but only their corresponding eigenvalues are shrunk towards their mean.
% This is necessary since eigenvalues estimated from a finite data sample
% are too dispersed w.r.t. the ground truth.
% - In summary, the new covariance matrix is both more accurate (!) and
% better conditioned than than the sample covariance matrix.
%
%   Input parameters:
%       X [DOUBLE]: a data matrix (observations x variables)
%
%   Output parameters:
%       Rxx [DOUBLE]: regularized covariance matrix (variables x variables)
%
% [1] Ledoit, Olivier, and Michael Wolf. "A well-conditioned estimator for
% large-dimensional covariance matrices." Journal of multivariate analysis
% 88.2 (2004): 365-411.

% Author: Simon Van Eyndhoven, KU Leuven, ESAT

%% Initialization
% Find the size of the data
[nobs,nvar] = size(X);

% Perform mean subtraction
X = bsxfun(@minus,X,mean(X,1));

%% Compute the sample covariance matrix
assert(nobs>1) % at least two observations needed
S = (1/(nobs-1))*(X'*X);

%% Compute the shrinkage parameter and the weights
% note: adjusted definition of norm (cfr. section 2 in [1]):
% ||X||^2 = trace(X*X^T)/size(X,2) (and not trace(X*X^T))
m = trace(S)/nvar; % cfr. lemma 3.2 in [1]
d2 = (norm(S-m*eye(nvar),'fro').^2)/nvar; % cfr. lemma 3.3 in [1]
b2 = min(calcbbar2(X,S),d2); % cfr. lemma 3.4 in [1]
a2 = d2 - b2; % cfr. lemma 3.5 in [1]

%% Compute the regularized covariance
Rxx = b2/d2 * m * eye(nvar) + a2/d2 * S; % cfr. eq. (14) in [1]
end

function bbar2 = calcbbar2(X,S)
% Compute coefficient bbar^2 according to lemma 3.4 in [1]. Here, the
% expression is not computed explicitly (i.e. via a for loop), but using a
% significantly faster procedure using array operations.

[nobs,nvar] = size(X);
rownorms = sum(X.^2,2);
term11 = X'*bsxfun( @times , X , rownorms ); % first squared term
term12 = -2*S*(X'*X); % cross-term
term22 = nobs*(S*S'); % second squared term

bbar2 = trace(term11 + term12 + term22)/(nvar*nobs^2);
end