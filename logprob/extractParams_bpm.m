function [a rho pi theta] = extractParams_bpm(omega, K, N, D)

% Shakir - script to convert vector of all params to constituent matrices

a = omega(1);

rho = omega(2:K+1);

tmp = omega(K+2:N*K+K+1);
pi = reshape(tmp,N,K);

tmp = omega((N*K+K+2):(N*K+K*D+K+1));
theta = reshape(tmp, K, D); % dims consistent with size at init
