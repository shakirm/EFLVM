function data = genpartmembin(n,alpha,b)

 close all
% inputs:
% n - number of dp to be generated
% alpha - 1xk vector of dirichlet hypers
% b - scalar hyper for exponential distribution
%
% Katherine

setSeed(1)
b=2;
k=length(alpha);
p = dirichlet_sample(alpha);
a = gamrnd(repmat(p, n, 1),1);
pi=zeros(n,k);

d=32;
theta=zeros(k,d);
XX=zeros(n,d);

% generate from prior
aa=repmat(0.2,1,d); 
bb=repmat(0.4,1,d);
for ii=1:k
    theta(ii,:)=betarnd(aa,bb);
end;

natpam=log(theta./(1-theta));
for ii=1:n
    pi(ii,:)=dirichlet_sample(a(ii,:).*p);
    upnatpam=ones(1,d);
    upnatpam=pi(ii,:)*natpam;  
    PPP=1./(exp(-upnatpam)+1);
    XX(ii,:)=binornd(1,PPP);
end

% reorder data to make easy to interpret
tmp=pi*[1 2 3]';
[ii jj]=sort(tmp);
XX=XX(jj,:);
pi=pi(jj,:);

data.truePi = pi;
data.X = XX;
data.theta = theta;
data.alpha = alpha;
data.b = b;
data.a = a;
% data.miss = 1:numel(XX); % train and test data the same

plot = 0;
if plot
    imagesc(XX); colormap gray;
    %plot(XX(:,1),XX(:,2),'.');
    
    figure(3);
    imagesc(theta); colormap gray;
    
    figure(4);
    imagesc(pi); colormap gray;
    % colorbar;
    
    imagesc(pi*pi');
end;
