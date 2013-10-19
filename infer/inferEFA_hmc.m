function postDist = inferEFA_hmc(data, options, K)
% Wrapper function for EFA code
% This specifies the initial sample and run the sampler, returning the
% result and other statistics of the samper behaviour in postDist.
%

% Shakir

% Energy function
logprob = @efaJointProb;
options.model = 'efa';

%% Data
[N,D] = size(data.X);
dims.K = K;
dims.N = N;
dims.D = D;
hypers.dims = dims; % Latent factors

%% Hyperparameters and Initialisation
alpha = 2.1; 
beta = 3.1;
a = repmat(2, 1, D); 
b = repmat(5, 1, D);
lambda = [a' b'];
m = zeros(K,1); % dim Kx1
C = 0.5; S = C*eye(K);

hypers.a = a;
hypers.b = b;
hypers.lambda = lambda;
hypers.alpha = alpha;
hypers.beta = beta;
hypers.m = m;
hypers.S = S;

% Initial sample
V = randn(N,K);
Theta = rand(K, D);
mu = randn(K,1);
Sigma = 1./gamrnd(alpha,1/beta,1,K); 
initVec = [mu(:); log(Sigma(:)); V(:); Theta(:)];
options.initVec = initVec;

%% Run HMC
tic;
[samples, stats] = hmc(logprob, data, hypers, options);
postDist.time = toc;

%% Store results
postDist.samples = samples;
postDist.energy = stats.energy;
postDist.accptRate = sum(stats.accept)/length(stats.accept);
postDist.acceptProb = stats.acceptProb;




