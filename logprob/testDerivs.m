function testDerivs(model)
% Check that the derivatives are computed correctly using finite
% differences
%
% Shakir, October 2012

setSeed(10);
epsilon = 0.0001;
verbose = 1;

% Test Data
N = 10;
D = 3;
K = 2;
dims.N = N; dims.D = D; dims.K = K;
X = rand(N,D) > 0.5;
TRANSFORM = 0;

switch model
    case 'bpm'
        testFn = @bpmJointProb;
        
        hypers.lambda = rand(D,2);
        hypers.alpha = 1.1;
        hypers.b = 1.1;
        hypers.dims = dims;
        L = D*K + N*K + K +1;
        
        a=rand;
        rho=rand(K,1); rho=rho/sum(rho);
        pie=rand(N,K); pie=pie./repmat(sum(pie,2),1,K);
        theta=rand(K,D);
        
        w=[a; rho(:); pie(:); theta(:)]; % constrained
        
        % Run and check that d is in order of 1e-9
        [d dy dh] = jf_checkgrad({testFn, hypers, X, K, TRANSFORM},w, epsilon,[],verbose);
        
    case 'efa'
        testFn = @efaJointProb;
        
        hypers.lambda = rand(D,2);
        hypers.m = zeros(K,1);
        hypers.S = eye(K);
        hypers.alpha = 1.1;
        hypers.beta = 1.1;
        hypers.dims = dims;
        L = N*K + N*D + 2*K;
        
        V = randn(N,K);
        Theta = rand(K, D);
        mu = randn(K,1);
        Sigma = rand(K,1);
        w = [mu(:); Sigma(:); V(:); Theta(:);];
        
        params.hypers = hypers;
        params.data.X = X;
        
        % Run and check that d is in order of 1e-9
        [d dy dh] = jf_checkgrad({testFn, params, TRANSFORM},w, epsilon,[],verbose);
        
    case 'bayesLogReg'
        testFn = @bayesLogReg;
        
        % Test data
        N = 5; D = 3;
        w = randn(D+1,1);
        params.data.X = randn(N,D);
        params.data.Y = double(sigmoid([ones(1,N); params.data.X']'*w) > 0.5);
        params.hypers.alpha = 10;
        
        % Run and check that d is in order of 1e-9
        [d dy dh] = jf_checkgrad({testFn, params},w, epsilon,[],verbose);
        
    case 'stochasticVol'
        testFn = @stochasticVol;
        % Test data
        T = 5;
        params.Y = randn(T,1);
        x = randn(T,1);
        phi = -0.1;
        beta = 1;
        sigma = 1;
        
        % ----1. Test derivs wrt latents
        fprintf('\n===Derivs wrt latents x \n');
        params.phi = phi; % abs(phi) < 1
        params.beta = beta;
        params.sigma = sigma; % sigma > 0
        TRANSF = 0; % whether to do transforms on params
        
        params.mode = 'latents';
        varin = x(:);
        
        % Run and check that d is in order of 1e-9
        [d dy dh] = jf_checkgrad({testFn, params,TRANSF},varin, epsilon,[],verbose);
        
        % ----2. Test derivs wrt params
        fprintf('\n===Derivs wrt params [phi beta sigma] \n');
        params.mode = 'params';
        params.x = x;
        varin = [phi; beta; sigma];
        TRANSF = 1;
        
        [d dy dh] = jf_checkgrad({testFn, params,TRANSF},varin, epsilon,[],verbose);
        
        fprintf('\n===Derivs wrt all [x phi beta sigma] \n');
        params.mode = 'all';
        varin = [x; phi; beta; sigma];
        TRANSF = 1;
        
        [d dy dh] = jf_checkgrad({testFn, params,TRANSF},varin, epsilon,[],verbose);
        
    case 'logGaussCox'
        testFn = @logGaussCox;
        
        % Test data
        N = 10;
        params.Y = poissrnd(1,N,1);
        x = randn(N,1);
        q = randn(N);
        params.Sigma = q*q'/N;
        params.cholSigma = chol(params.Sigma);
        params.mu = log(10) - 1.91/2;
        params.m = 1/N^2;
        
        fprintf('\n===Derivs wrt latents x \n');
        varin = x(:);
        
        % Run and check that d is in order of 1e-9 or lower
        [d dy dh] = jf_checkgrad({testFn, params},varin, epsilon,[],verbose);
        
    otherwise
        error('Unknown model to test derivatives');
end;

