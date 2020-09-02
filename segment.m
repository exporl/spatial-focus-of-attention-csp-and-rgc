function segmented = segment(X,segSize)
% SEGMENT Segment given signal X, with the time the first dimension, into
% segments of the given segSize (in samples) and make new trials out of it.
%
%   Input parameters:
%       X [DOUBLE]: EEG tensor (time x channel x band x trial)
%       segSize [INTEGER]: size of each segment in samples
%
%   Dependency:
%       Tensorlab (https://www.tensorlab.net/)
%

% Authors: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

segmented = segmentize(X,'Segsize',segSize);
segmented = permute(segmented,[3,4,1,2,5]);
segmented = reshape(segmented,[size(segmented,1),size(segmented,2),size(segmented,3),size(segmented,4)*size(segmented,5)]);
end