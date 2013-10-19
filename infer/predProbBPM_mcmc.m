function [nlp rmse_test pHat] = predProbBPM_mcmc(data, samples, K, burnin, thin)
% compute predictive probability (on test data) for BPM model
% e.g., predProbBPM_mcmc(Xtest, samples, K)
%
% Test data are elements of the observed data, data.X that are missing and
% set to -1. Data.miss contains the the masking matrix of elements that are 
% missing. These lements of data.miss set to 1 are the test data and we use 
% these indices to compute the predictive probability.
%
% Shakir

% FIX HERE - WHAT IS THE FORMAT OF THE SAMPLES
% MAKE CONSISTEMT WITH EFA FILE

X = data.X;

% samples = [ignore1 ignore2 a rho phi theta]
[N,D] = size(X);
[nsamples, ~] = size(samples);

pairwise = zeros(N,D);
ppp = zeros(N,D);
startPos = burnin;
endPos = nsamples;
len = endPos - startPos + 1;
ct = 0;
for i = startPos:thin:endPos
   ct = ct +1;
   [a rho pi theta] = extractParams_bpm(samples(i,:), K, N, D);
   pairwise = pi*theta;
   ppp = ppp + pairwise;
end;
eta = ppp./ct; % For calc mean without storing
pHat = sigmoid(eta); % For binary data only

% Neg log prob on test elements
miss = data.miss;
nlp = -sum(sum(log(pHat(miss))))/sum(sum(miss));

% RMSE
rmse_test = sqrt(mean((pHat(X ~= -1) - X(X ~= -1)).^2));

%[pHat(Xtest>=0) Xtest(Xtest>=0)]
