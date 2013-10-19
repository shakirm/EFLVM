function [nlp rmse_test pHat] = predProbEFA_mcmc(data, samples, K, burnin, thin)
% compute predictive probability (on test data) for EFA model
% e.g., predProbBPM_mcmc(data, samples, K)
%
% Test data are elements of the observed data, data.X that are missing and
% set to -1. Data.miss contains the the masking matrix of elements that are 
% missing. These lements of data.miss set to 1 are the test data and we use 
% these indices to compute the predictive probability.
%
% Shakir

% samples = [ignore1 ignore2 a rho phi theta]
X = data.trueX;
trueX = data.trueX;
[N,D] = size(X);
[nsamples, ~] = size(samples);

pairwise = zeros(N,D);
ppp = zeros(N,D);
endPos = nsamples;
startPos = burnin;
len = endPos - startPos + 1;
ct = 0;
for i = startPos:thin:endPos
   [V, Theta, ~, ~] = extractParams(samples(i,:),D, N, K);
   pairwise = V*Theta;
   ppp = ppp + pairwise;
   ct = ct+1;
end;
eta = ppp./ct; 
pHat = sigmoid(eta); % binary data

% Calc neg log prob on test elements
logP1 = log2(pHat);
logP2 = log2(1 - pHat);
% idx = find(data.miss); % Get indices of testing data
idx = data.miss;
% aa = idx(1:10)
% [X(aa)' pHat(aa)'>0.5]
% pause

pp = sum(sum(X(idx).*logP1(idx) + (1 - X(idx)).*logP2(idx))); % Calc the sum
nlp = -pp;

% Calc RMSE
rmse_test = sqrt(mean((pHat(idx) - trueX(idx)).^2));


