function [logp grad] = bayesLogReg(weight, params)
% - weight in format of [bias weights]
% - To check the gradients use: testDerivs('bayesLogReg')

weight = weight(:);
X = params.data.X'; % covariates (DxN)
y = params.data.Y; % labels, (Nx1)
alpha = params.hypers.alpha; % noise param (scalar)
[D,N] = size(X); % mean corrected covariates
P = size(y,2);

X = [ones(P,N); X]; % append for bias (D+1 x N)
eta = X'*weight;

%% --- Log probability
Term = [];
Term(1) = LogNormPDF(weight, zeros(D+1,1),alpha); %logGauss prior
Term(2) = y'*eta - sum(log(1+exp(eta))); % Lik = y'eta - log(1+exp(eta))
logp = sum(Term);

%% --- gradient of log p(x)
if nargout == 2
    dw = X*(y - sigmoid(eta)) - (1/alpha)*weight; 
    grad = dw;
end;
