function [logp grad] = stochasticVol(varin, params, TRANSF)
%
% Format of varin [x phi beta sigma]
%
% Shakir: Adapted from code by Girolami and Calderhead. Bugs in derivatives fixed.
%

if nargin<3, TRANSF=0; end;

%Get data
y = params.Y(:);
T = length(y);
[r,c] = size(varin);
if min(r,c) == r
    varin = varin';
end;

switch params.mode
    case 'latents'
        x = varin; % latent log-vol
        beta = params.beta; % scaling (instantaneous vol)
        phi = params.phi;
        sigma = params.sigma;
        
    case 'params'
        x = params.x(:);
        phi = tanh(varin(1)); %
        beta = varin(2); % scaling (instantaneous vol)
        sigma = exp(varin(3)); % var of log-vol
        
    case 'all'
        x = varin(1:T); % latent log-vol
        phi = tanh(varin(end-2)); %
        beta = varin(end-1); % scaling (instantaneous vol)
        sigma = exp(varin(end)); % var of log-vol
end;

if TRANSF 
    % for use with testDerivs and checkgrad
    % use only for 'params' or 'all'
    phi = varin(end-2); %
    sigma = varin(end); % var of log-vol
end;

% Log probability
Term = [];
Term(1) = -sum(x/2) - T*log(beta) - sum((y.^2)./(2*beta^2*exp(x)) ); % p(y|x, beta)
Term(2) = + 0.5*log(1-phi^2) - log(sigma) - x(1)^2*(1-phi^2)/(2*sigma^2); % p(x_1)
Term(3) =  - (T-1)*log(sigma) - sum((x(2:end)-phi*x(1:end-1)).^2./(2*sigma^2)); % p(x_t | x_t-1, phi, sigma)
Term(4) = log(sigma) + log(1-phi^2); % Jacobian for transforms on sigma and phi

if strcmp(params.mode,'params') || strcmp(params.mode,'all')
    Term(5) = -(beta); % p(beta)
    Term(6) = -0.5/(2*sigma^2) - 6*log(sigma^2) + log(sigma); % p(sigma) ?don't know about last term here
    Term(7) = +19*log((phi+1)/2) + 0.5*log((1-phi)/2); % p(phi)
end;

logp = sum(Term);

% gradient of log p(x) (if needed)
if nargout == 2
    grad = [];
    if strcmp(params.mode,'all') || strcmp(params.mode,'latents')
        s = -0.5 + (y.^2)./(2*beta^2*exp(x));
        d_1 = (1/sigma^2)*(x(1) - phi*x(2));
        d_2 = (1/sigma^2)*(x(end) - phi*x(end-1));
        r = [d_1; -(phi/sigma^2)*(x(3:end) - phi*x(2:end-1)) + (1/sigma^2)*(x(2:end-1) - phi*x(1:end-2)); d_2];
        dx = s -r; % correct
        
        grad = [grad(:); dx(:)];
    end;
    
    if strcmp(params.mode,'all') || strcmp(params.mode,'params')      
        dphi = (-phi + ((1-phi^2)*phi*x(1)^2)/sigma^2 ... % p(x1)
            + (1-phi^2)*sum(x(1:end-1).*(x(2:end)-phi*x(1:end-1)))/sigma^2 ... % p(xt | xt-1)
            + 19*(1-phi) -0.5*(1+phi) ... % p(phi)
            - 2*phi); % ... % jacobian
        
        % can you just leave this part out (as they do in Giro?)
        if TRANSF
            % for testing using orig grad
            dphi = dphi/((1-phi^2));
        end;
        
        dbeta = -T/beta + sum((y.^2)./(beta^3*exp(x))) - 1;
        
        dsigma = -(T-1) + x(1)^2*(1-phi^2)/(sigma^2) ...
            + sum( ((x(2:end)-phi*x(1:end-1)).^2)/(sigma^2) ) ...
            + 0.5/sigma^2 - 11;
        
        grad = [grad(:); dphi(:); dbeta(:); dsigma(:)];
    end;
    
    grad = grad';

end;