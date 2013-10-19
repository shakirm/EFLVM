function [samples, stats] = hmc(logprob, data, hypers, options)
%
% HMC Hybrid Monte Carlo sampling method
% General-purpose HMC sampler. Requires a function logprob that return the
% joint probability and its derivatives (do not return negative).
%
% [samples, stats] = hmc(LOGPROB, DATA, HYPERS, OPTIONS
%
% LOGPROB: function that return the log joint probability of the model and
% the first derivatives with respect to all parameters to be sampled.
% DATA: Struct with the data (e.g. DATA.X and DATA.Y for regression.
% HYPERS: Struct with any fixed hyperparameters of the model
% OPTIONS: Struct with options controllinng HMC. See hmc.m
%
% CONSTRAINED VARIABLES
% For models with constrained variables (e.g., variances, or probabilities),
% also specify a function called TRANSFROM that allows the samples (which 
% are produced in uncnstrained parameter space to be returned in the 
% original, constrained paramter space. See transform.m
%
% MISSING DATA
% To handle missing data, pass a masking matrix of elements that are
% abserved in the data struct and use this to only include observed data in
% when comouting the joint probability using LOGPROB.
%
% SEE ALSO: bayesLogRreg, efaJointProb, transform, getConfigEFLVM
%

% Shakir, October 2012

[stepSize, nSamples, nLeaps, initVec, mass, display] = myProcessOptions(options, 'stepSize', 1e-3, ...
		'nSamples', 200, 'nLeaps',50, 'initVec', [], 'mass', 1, 'display',1);

if ~isfield(options,'model')
    options.model = '';
end;
if ~isfield(hypers,'dims')
    hypers.dims.N = size(data.X,1);
    hypers.dims.D = size(data.X,2);
end;
if ~isfield(options,'initVec')
    error('Must pass an initial vector using options.initVec');
end;

maxLeaps = nLeaps; 
maxStepSize = stepSize; 
omega = initVec;
nParams = length(initVec);
dims = hypers.dims;
params.data = data;
params.hypers = hypers;

[logp, grad] = logprob(omega, params);

for i = 1:nSamples
    p = randn(nParams,1); 
    H = p'*p/2 - logp; 
    
    omegaNew = omega; 
    gradNew = grad;
    
    % Randomise leapfrogs/stepsize
    numLeaps = ceil(rand*maxLeaps); 
    stepSize = maxStepSize; % fixed epsilon
    
    for t = 1:numLeaps
        p = p + stepSize*gradNew/2; % half step in p
        if sum(isnan(p)) > 0, break; end
        
        omegaNew = omegaNew + stepSize*p; % make a step
        [logpNew, gradNew] = logprob(omegaNew, params);
        p = p + stepSize*gradNew/2; % half step in p
    end;
    
    accept = 0;
    Hnew = p'*p/2 - logpNew;
    dH = Hnew - H;
    accVal = rand;
    if (accVal < min(1,exp(-dH)))
        accept = 1;
        grad = gradNew;
        logp = logpNew;
        omega = omegaNew;
    end;
    
    if display
        fprintf('%3d, Energy = %4.2f, Accept = %d, Stepsize = %2.2e, L=%2.0f\n',i,-logp,accept, stepSize, numLeaps);
    end;
    
    % Save results
    samples(i,:) = transform(omega,options.model,dims);
    energy(i) = -logp;
    acceptProb(i) = -dH;
    acc(i) = accept;
end;

fprintf('\nHMC Complete\n');
stats.energy = energy;
stats.acceptProb = acceptProb;
stats.accept = acc;
