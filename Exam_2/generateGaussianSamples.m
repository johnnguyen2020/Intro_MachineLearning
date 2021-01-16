function [data, classIndex] = generateGaussianSamples(mu, sigma, nSamples, prior)
% Function to simulate data from k Gaussian densities (1 for each class) in d dimensions.
%
% INPUTS:
%   mu          - cell with the class dependent d-dimensional mean vector
%   sigma       - k-by-1 cell with the class dependent d-by-d covariance matrix
%   nSamples    - scalar indicating number of samples to be generated
%   prior       - k-by-1 vector with class dependent mean
%
% OUTPUTS:
%   data        - nSamples-by-d array with the simulated data distributed along the rows
%   classIndex  - vector of length nSamples with the class index for each datapoint

if sum(prior) ~= 1
    error('Priors should add to one!');
end

% First, sample the respective class indexes. We can do this by generating uniformly
% distributed numbers from 0 to 1 and using thresholds based on the prior probabilities 

classTempScalar = rand(nSamples, 1);
priorThresholds = cumsum([0; prior]);
nClass = numel(mu);
data = cell(nClass, 1);
classIndex = cell(nClass, 1);

for idxClass = 1:nClass
    nSamplesClass = nnz(classTempScalar>=priorThresholds(idxClass) & classTempScalar<priorThresholds(idxClass+1));
    % Generate samples according to class dependent parameters
    data{idxClass} = mvnrnd(mu{idxClass}, sigma{idxClass}, nSamplesClass);
    % Set class labels
    classIndex{idxClass} = ones(nSamplesClass,1) * idxClass; 
end

data = cell2mat(data);
classIndex = cell2mat(classIndex);
end