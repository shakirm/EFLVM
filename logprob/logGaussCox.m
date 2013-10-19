function [logp grad] = logGaussCox(x, params)
%
% Gradient of log Gaussian Cox point process for latent Gaussian x only.
%
% Shakir: Adapted from code by Girolami and Calderhead.
% To add hyperparameters mu, sigma, beta later
%

% Get data and params
y = params.Y(:);
x = x(:);
D = length(y);
mu = params.mu*ones(D,1);
Q = params.cholSigma;
%Sigma = data.Sigma;
m = params.m;

%Q = chol(Sigma);
z = (x - mu);
alpha = Q'\z;

% Log probability
Term = [];
Term(1) = y'*x - sum(m*exp(x)) ; % Poiss likelihood
Term(2) = -0.5*(alpha'*alpha); % Gauss Prior

logp = sum(Term);

% gradient of log p(x) (if needed)
if nargout == 2
    grad = [];
    dx = y - m*exp(x) - Q\alpha;    
    grad = [grad(:); dx(:)];
    
    grad = grad';
end;



