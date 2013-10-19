function exptEFLVM(dataName, model, K, seed)
%
% e.g., exptEFLVM('votes', 'bpm',3)
% Experiment file used to run EXFA and BPM.
% Shakir


close all; clear global;
if nargin < 4, seed = 1; end;

[dataDir, outDir] = setupDir;
fileName = sprintf([outDir '/%s/%s_%s_%d_%d'], dataName,model, 'hmc', K, seed);

% Load Data
data = getBinData(dataName, dataDir);

% Get config settings
options = getConfigEFLVM(dataName, model);
burnin = options.burnin;
thin = options.thin;
saveOut = options.saveout;

setSeed(seed);

% Set functions
switch model
    case 'bpm'
        inferFun = @inferBPM_hmc;
        predFun = @predProbBPM_mcmc;
    case 'efa'
        inferFun = @inferEFA_hmc;
        predFun = @predProbEFA_mcmc;
end;

% run HMC
postDist = inferFun(data, options, K);

% Save
if saveOut
    fprintf('Saving as %s\n',fileName);
    save(fileName, 'postDist', 'seed');
end

% test error
if isfield(data,'miss')
    % only test if testing data exists
    [teErr, rmse, pHat] = predFun(data, postDist.samples, K, burnin, thin);
    if saveOut
        save(fileName,'teErr', 'rmse','-append');
    end;
end;






