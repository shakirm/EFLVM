function postDist = inferBPM_hmc(data, options, K)
% Wrapper function for BPM code
% This specifies the initial sample and run the sampler, returning the
% result and other statistics of the samper behaviour in postDist.
%
% Shakir

% Energy function to use
logprob = @bpmJointProb;
options.model = 'bpm';

%% Data
X = data.X;
[N,D] = size(X);
dims.K = K;
dims.N = N;
dims.D = D;
hypers.dims = dims;

%% Hyperparameters and Initialisation
a = repmat(2, 1, D);
b = repmat(5, 1, D);
lambda = [a' b']; % This is reset to mean of data in code so not used right now
beta = 1.1; % Exponential hyper distr a ~ p(a| beta)
alpha = 1.1; % Dirichlet hyper rho ~ p(rho | alpha)
hypers.alpha = alpha;
hypers.beta = beta;
hypers.lambda = lambda;

a=rand;
rho=rand(K,1); rho=rho/sum(rho);
pie=rand(N,K); pie=pie./repmat(sum(pie,2),1,K);
theta=rand(K,D);
initVec = [log(a); log(rho(:)); log(pie(:)); theta(:)]; % unconstrained
options.initVec = initVec;

%% Run HMC
tic;
[samples, stats] = hmc(logprob, data, hypers, options);
postDist.time = toc;

%% Collect samples
postDist.samples = samples;
postDist.energy = stats.energy;
postDist.acceptProb = stats. acceptProb;
postDist.accptRate = stats.accept/sum(stats.accept);

