function [logp grad] = efaJointProb(omega, params, TRANSFORM)
%
% Joint prob and energy for Binary BPM.
% All mising data is encoded as -1 in the params.data.X.
%
% --USAGE
% [logp grad] = efaJointProb(omega, params, TRANSFORM)
%
% OMEGA: vector with a cuurent sample to evaluate. Contains vectorisations 
% of all paramters that are sampled.
% PARAMS: Struct with entries PARAMS.DATA, PARAMS.HYPERS, and PARAMS.DIMS/
% For factor models such as BPM and EFA, the latent dimensionality K is
% passed throug PARAMS.DIMS. PARAMS.DATA is a struct with all required data
% (such as X and Y for regression models). 
% TRANSFORM: This is only used for testing the derivatives.
%
% --CONSTRAINED VARIABLES
% If the model constains constrained variables (e.g., variances or
% proabilities), ensure that the joint probability contains the Jacobian of
% the transformation to an uncnstrained varaible and that the chain rule is
% applied correctly. This model, EFA, is a good example of how to handle
% such constraints.
%
% --MISSING DATA
% This log probability handles missing data, which is encoded using -1 in
% params.data.X. We only include observed data in computation.
%

% Shakir, October 2012

if nargin < 3, TRANSFORM = 1; end;
hypers = params.hypers;
X = params.data.X;
K = hypers.dims.K;

% unpack the vector
[N, D] = size(X);
[V Theta Sigma mu] = extractParams(omega, D, N, K);
lambda = hypers.lambda;
m = hypers.m;
S = hypers.S;
alpha = hypers.alpha;
beta = hypers.beta;

if TRANSFORM
    % Transform back to original, constrained variable
    Sigma = exp(Sigma);
end;

invS = diag(1./diag(S));
invSigma = 1./Sigma; % Sigma is diagonal
lnSigma = log(Sigma); % log of sigma used by invGamma prior
sumLnSigma = sum(lnSigma); % sum(ln(sigma))
natparams = V*Theta;

Terms = [];
Terms(1) = sum(sum(natparams.*(X==1))) - sum(sum(((X==0) + (X==1)).*log(1 + exp(natparams))));
Terms(2) = -sum(sum(log(1+ exp(-Theta))*lambda(:,1))) - sum(sum(log(1+ exp(Theta))*lambda(:,2)));
Terms(3) =  K*(sum((gammaln(lambda(:,1) + lambda(:,2)) ...
   - gammaln(lambda(:,1)) - gammaln(lambda(:,2))))) ...
    -sum(sum(Theta - 2.*log(1 + exp(Theta))));
Terms(4) = -N*K/2*log(2*pi) - N*0.5*log(det(diag(Sigma)));

U = repmat(mu',N,1); 
Y = V - U; 
val = sum(diag(Y*diag(invSigma)*Y'));
Terms(5) = -0.5*val;

Terms(6) = -K/2*log(2*pi) - 0.5*logdet(S) - 0.5*(mu - m)'*invS*(mu - m); 
Terms(7) = + K*alpha*log(beta) - K*gammaln(alpha) + alpha*sumLnSigma;
Terms(8) = -beta*sum(Sigma);

logp = sum(Terms);

% Compute gradient if required
if nargout == 2 
    iS = diag(invSigma); % make it a diagonal matrix
    f1 = Y'*Y;
    expRatio = 1./(1 + exp(Theta)); % dim KxD
    expTerm = exp(Theta);% dim KxD
    
    dmu = sum(diag(invSigma)*Y',2) - invS*(mu - m);
    
    dSigmak = -N*0.5.*invSigma + (alpha*invSigma - beta) ...
        + diag(0.5*iS*f1*iS);
    
    dVnk = (X==1)*Theta' - sigmoid(natparams)*Theta' - Y*diag(invSigma);
    
    dThetakd = repmat(lambda(:,1)',K,1).*expRatio ...
        - repmat(lambda(:,2)',K,1).*expRatio.*expTerm ...
        - (1 - 2.*expRatio.*expTerm) ...
        + V'*(X==1) - V'*(sigmoid(natparams).*((X==1) + (X==0)));
    
    % Derivaive Chain rule
    ndSigmak = Sigma.*dSigmak;
    
    if ~TRANSFORM, ndSigmak = dSigmak; end; % Used for Deriv testing
    
    % Final gradient
    grad = [dmu(:); ndSigmak(:); dVnk(:); dThetakd(:)];
    
end;

