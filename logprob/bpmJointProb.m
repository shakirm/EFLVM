function [logp grad] = bpmJointProb(omega, params, TRANSFORM)
%
% Joint prob and energy for for Binary BPM model. All mising data is
% encoded as -1 in the params.data.X.
%
% --USAGE
% [logp grad] = bpmJointProb(omega, params, TRANSFORM)
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
% applied correctly. This model, BPM, is a good example of how to handle
% such constraints.
%
% --MISSING DATA
% This log probability handles missing data, which is encoded using -1 in
% params.data.X. We only include observed data in computation.
%
% SEE ALSO: hmc, efaJointProb, transform

% Shakir and Katherine, 2012


if nargin<3, TRANSFORM = 1; end;
hypers = params.hypers;
X = params.data.X;
K = hypers.dims.K;

% Extract params
[N, D] = size(X);
a = omega(1);
rho = reshape(omega(2:K+1),K,1);
pie = reshape(omega(K+2:(N*K+K+1)),N,K);
theta = reshape(omega(N*K+K+2:length(omega)),K,D);

lambda = hypers.lambda;
alpha = hypers.alpha;
b = hypers.beta;

if TRANSFORM
    a = exp(a);
    rho = exp(rho);
    rho = rho./sum(rho);
    pie = exp(pie);
    pie = bsxfun(@times,pie,1./sum(pie,2)); % normalise rows
end;

% Log Joint Probability
the1= -log(1+exp(-theta));
the2= -theta + the1;
natparam = pie*theta;

Terms = [];
Terms(1) = gammaln(sum(alpha))-sum(gammaln(alpha))+sum((alpha-1).*log(rho));

Terms(2) = +log(b) - b*a + N*(gammaln(sum(a*rho)) - sum(gammaln(a*rho)));

Terms(3) = +sum(log(pie)*(a*rho-1)) + sum(sum((pie*theta).*(X==1))) ...
    + sum(sum(-((X==0) + (X==1)).*log(exp(natparam) + 1))) ...
    + sum(the1*lambda(:,1) + the2*lambda(:,2));

Terms(4) = sum(sum(-log(exp(theta)+1))) ...
    + K*(sum(gammaln(lambda(:,1)+lambda(:,2)+3)-gammaln(lambda(:,1)+1)-gammaln(lambda(:,2)+2))) ...
    + sum(sum(log(pie))) + sum(log(rho)) + log(a);

logp = sum(Terms);

% Compute gradient if required
if nargout == 2
    thesm = sigmoid(theta);
    da = -b + N*(digamma(sum(a*rho))*sum(rho) - sum(digamma(a*rho)'*rho)) ...
        + sum(log(pie)*rho) + 1/a;
    
    drho=(alpha-1)./rho + N*(repmat(digamma(sum(a*rho))*a,K,1) ...
        - digamma(a*rho)*a) + sum(a*log(pie),1)' + 1./rho;
    
    Gp = -sigmoid(pie*theta).*((X==1)+(X==0));
    
    dpi = repmat((a*rho-1)',N,1)./pie + (X==1)*theta'+ Gp*theta'+1./pie;
    
    dtheta = pie'*(X==1) + pie'*Gp + repmat(lambda(:,1)',K,1) ...
        - repmat((lambda(:,1) + lambda(:,2))',K,1).*thesm - sigmoid(theta);
    
    % Derivative Chain Rule
    nda=da*a;
    ndrho=(drho.*rho) - (drho'*rho)*rho; 
    for n=1:N
        ndpi(n,:)=(dpi(n,:).*pie(n,:)) - (dpi(n,:)*pie(n,:)')*pie(n,:); % slow
    end;
    
    if ~TRANSFORM, nda = da; ndrho = drho; ndpi = dpi; end; % For deriv test only
    
    % Final gradient
    grad = [nda; ndrho(:); ndpi(:); dtheta(:)];
       
end;