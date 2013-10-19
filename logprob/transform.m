function sample = transform(omega, model, dims)
%
% Transforms vector of samples with transformed variables to original
% space.
%
% Shakir

switch model
    case 'efa'
        sample = omega;
        sample(dims.K+1:2*dims.K) = exp(sample(dims.K+1:2*dims.K));
        
    case 'bpm'
        a = omega(1);
        rho = reshape(omega(2:dims.K+1),dims.K,1);
        pie = reshape(omega(dims.K+2:(dims.N*dims.K+dims.K+1)),dims.N,dims.K);
        theta = reshape(omega(dims.N*dims.K+dims.K+2:length(omega)),dims.K,dims.D);
        a = exp(a);
        rho = exp(rho);
        rho = rho./sum(rho);
        pie = exp(pie);
        pie = bsxfun(@times,pie,1./sum(pie,2)); % normalise rows
        sample = [a(:); rho(:); pie(:); theta(:)];
        
    otherwise
        % No transformation
        sample = omega;
end;
