function plotEFLVM(experiment)
% Recreate plots in chapter
% Shakir, August 2012

[dataDir, outDir] = setupDir;
print = 0; % create pdfs

switch experiment
    case 'synth-efa'
        %% 1. Binary Synthetic data
        dataName = 'synth';
        model = 'efa';
        K = 3;
        seed = 1;
        
        % Get directories
        [dataDir, outDir] = setupDir;
        fileName = sprintf([outDir '/%s/%s_%s_%d'], dataName,model, 'hmc', K)
        
        % Run HMC and get Results
        if ~exist([fileName,'.mat'],'file')
            exptEFLVM(dataName,model,K, seed);
        end;
        res = load(fileName);
        
        % Load Data
        setSeed(res.seed);
        dat = getBinData(dataName, dataDir);
        [N,D] = size(dat.trueX);
        
        % Sample indices to plot
        figure;
        ix = [2, 100, 500, 1000,3000, 5000];
        % Reconstructions for various samples
        for i = 1:length(ix)
            [V, Theta , ~, ~] = extractParams(res.postDist.samples(ix(i),:), D, N, K);
            pStar = V*Theta;
            recon = 1./(1 + exp(-pStar))';
            subplot(3,4,i);
            imagesc(recon'> 0.5);
            title(sprintf('Sample %d',ix(i)),'FontSize',14,'FontWeight','bold');
            set(gca,'xticklabel',[],'yticklabel',[]);
            colormap gray;
        end;
        
        % Average recon
        pStar = zeros(N,D);
        rng = 3000:30:5000;
        for i = rng
            [V, Theta , ~, ~] = extractParams(res.postDist.samples(i,:), D, N, K);
            pStar = pStar + V*Theta;
        end;
        pStar = pStar./length(rng);
        recon = 1./(1 + exp(-pStar))';
        subplot(3,4,7);
        imagesc(recon'>0.5);
        title('Mean Reconstruction','FontSize',14,'FontWeight','bold');
        set(gca,'xticklabel',[],'yticklabel',[]);
        
        % Noise free data
        subplot(3,4,8);
        imagesc(dat.cleanX);
        title('True Data','FontSize',14,'FontWeight','bold');
        set(gca,'xticklabel',[],'yticklabel',[]);
        ylabel('Observations (N)');
        xlabel('Dimension (D)')
        
        % Energy (joint prob on train data)
        markPoints = [2, 100, 500, 1000,3000, 5000];
        rng = 1:5000;
        h = subplot(3,4,[9 10],'ActivePositionProperty','OuterPosition');
        semilogx(rng,res.postDist.energy,'b-','LineWidth',2);
        hold on
        semilogx(markPoints,res.postDist.energy(markPoints),'bo',...
            'LineWidth',2,'MarkerSize',10);
        ylim([2000 6000]);
        xlim([0 10000])
        set(gca,'ytick',[2000:1000:5000]);
        set(gca,'xtick',[10.^([0:4])])
        title('Log Joint Probability (Energy)','FontSize',14,'FontWeight','bold');
        xlabel('Sample','FontSize',14,'FontWeight','bold');
        grid on;
        
        % Predictive prob
        S = size(res.postDist.samples,1);
        teErr = []; rmse = [];
        for i = 1:S
            [teErr(i), rmse(i), ~] = predProbEFA_mcmc(dat, res.postDist.samples(i,:), K, 1, 1);
        end;
        h = subplot(3,4,[11 12],'ActivePositionProperty','OuterPosition');
        semilogx(rng,teErr,'b-','LineWidth',2)
        hold on
        semilogx(markPoints,teErr(markPoints),'bo',...
            'LineWidth',2, 'MarkerSize',12)
        ylim([0 800]);
        xlim([0 10000])
        set(gca,'ytick',[-200:200:600], 'Yticklabel',[-200:200:600]);
        set(gca,'xtick',[10.^([0:4])])
        title('Negative Log Predictive Probability (bits)','FontSize',14,'FontWeight','bold');
        xlabel('Sample','FontSize',14,'FontWeight','bold');
        grid on;
        
        % Create PDFs and move to correct directory
        printPDF('syntheticBin',outDir,print);
        
        
    case 'synth-bpm'
        %% 2. bpm with data generated from the model
        dataName = 'bpmmodel';
        model = 'bpm';
        K = 3;
        seed = 10;
        
        % Get directories
        [dataDir, outDir] = setupDir;
        fileName = sprintf([outDir '/%s/%s_%s_%d_%d'], dataName,model, 'hmc', K,seed)
        
        % Run HMC and get Results
        if ~exist([fileName,'.mat'],'file')
            exptEFLVM(dataName,model,K, seed);
        end;
        res = load(fileName);
        
        % Load Data
        dat = getBinData(dataName, dataDir);
        [N,D] = size(dat.X);
        
        [S,LL] = size(res.postDist.samples);
        gg = zeros(N,N);
        burnin = 2000;
        thin = 50;
        rng = burnin:thin:S;
        length(rng)
        for i = rng
            omega = res.postDist.samples(i,:);
            [~, ~, pi, ~] = extractParams_bpm(omega, K, N, D);
            gg = gg + pi*pi';
        end;
        GG = gg/length(rng);
        TP = dat.truePi*dat.truePi';
        
        r = 1; c = 2;
        % h1 = subplot(r,c,1);
        figure;
        subplot(1,2,1)
        imagesc(TP,[0 1]);
        title('True U_T','FontSize',14, 'FontWeight','bold')
        xlabel('Observations','FontSize',14, 'FontWeight','bold');
        ylabel('Observations','FontSize',14, 'FontWeight','bold');
        set(gca,'xticklabel',[],'yticklabel',[]);
        ht = colorbar('location','EastOutside');
        set(ht,'FontSize',12);
        axis square
        % Create PDFs and move to correct directory
        printPDF('bpmSynthetic_covPlots1',outDir,print);
        
        
        % h2 = subplot(r,c,2);
        figure;
        subplot(1,2,1)
        imagesc(GG,[0 1]);
        title('Inferred U_L','FontSize',14, 'FontWeight','bold')
        set(gca,'xticklabel',[],'yticklabel',[]);
        %         colormap gray;
        xlabel('Observations','FontSize',14, 'FontWeight','bold');
        ylabel('Observations','FontSize',14, 'FontWeight','bold');
        ht = colorbar('location','EastOutside');
        set(ht,'FontSize',12);
        axis square
        % Create PDFs and move to correct directory
        printPDF('bpmSynthetic_covPlots2',outDir,print);
        
        figure;
        subplot(1,2,1)
        diff = abs(GG(:) - TP(:));
        rng = 0.1:0.1:0.5;
        counts = histc(diff,rng);
        perc = counts./sum(counts);
        disp('Histogram of differences');
        disp([cumsum(perc)*100 rng']); % PPrint table instead
        bar(rng,perc,'FaceColor','b');
        title('Histogram of Differences |U_T - U_L|','FontSize',14, 'FontWeight','bold');
        set(gca,'xtick',rng,'xticklabel',rng, 'FontSize',12)
        xlim([0 0.55]);
        ylim([0 0.6]);
        set(gca,'xtick',rng);
        xlabel('Difference threshold','FontSize',14, 'FontWeight','bold');
        ylabel('% entries in bin','FontSize',14, 'FontWeight','bold')
        axis square;
        % Create PDFs and move to correct directory
        printPDF('bpmSynthetic_hist',outDir,print);
        
    case 'votes-bpm'
        %% 3. Senate votes for BPM
        dataName = 'votes';
        model = 'bpm';
        K = 2;
        seed = 1;
        
        % Get directories
        [dataDir, outDir] = setupDir;
        fileName = sprintf([outDir '/%s/%s_%s_%d'], dataName,model, 'hmc', K)
        
        % Run HMC and get Results
        if ~exist([fileName,'.mat'],'file')
            exptEFLVM(dataName,model,K, seed);
        end;
        res = load(fileName);
        
        % Load Data
        dat = getBinData(dataName, dataDir);
        [N,D] = size(dat.X);
        
        for i = 1:N
            if strfind(dat.lab{i},'(R-')
                RD(i) = 1;
            elseif strfind(dat.lab{i},'(D-')
                RD(i) = 2;
            elseif strfind(dat.lab{i},'(I-')
                RD(i) = 3;
            else
                RD(i) = 4;
            end;
        end;
        
        [S,LL] = size(res.postDist.samples);
        omega = res.postDist.samples(end,:);
        [a, rho, pi, theta] = extractParams_bpm(omega, K, N, D);
        [P, Q] = size(pi);
        [v, ix] = sort(pi(:,2));
        pl(:,1) = 1:100;
        pl(:,2) = v.*100;
        
        figure;
        plot(pl(:,1),v,'c','LineWidth',10)
        hold on
        plot(pl(:,1),v,'-','LineWidth',2,'Color',[0.8 0.8 0.8])
        hold on
        cols = {'r','b','m','k'};
        for i = 1:4:100
            tt = dat.lab(ix(i));
            cc = cols{RD(ix(i))};
            text(pl(i,1),v(i)-0.1*length(tt),tt,'Rotation',90,'Color',cc,'FontWeight','bold', 'FontSize',11);
            hold on;
        end;
        ylim([-0.1, 1.3])
        xlim([-1 101])
        grid on
        ylabel('Partial Membership','FontSize',14, 'FontWeight','bold');
        xlabel('Senator','FontSize',14, 'FontWeight','bold');
        title('Ideal Point Diagram','FontSize',14, 'FontWeight','bold');
        set(gca,'tickDir','out')
        
        % Show predictive probs
        pStar = sigmoid(pi*theta);
        for i = 1:100
            hh = -((dat.X ==1).*log2(pStar) + (dat.X == 0).*log2(1-pStar));
        end;
        hh2 = sum(hh,2);
        disp('Pred Prob (nats): mean min median max')
        disp([mean(hh2) min(hh2) median(hh2) max(hh2)]);
        disp('Outcome (nats)');
        disp(hh2(end));
        
        % Create PDFs and move to correct dplirectory
        printPDF('bpmVotes2',outDir,print);
        
    case 'votes-efa'
        %% 4. Senate votes for EFA
        dataName = 'votes';
        model = 'efa';
        K = 2;
        seed = 1;
        
        % Get directories
        [dataDir, outDir] = setupDir;
        fileName = sprintf([outDir '/%s/%s_%s_%d'], dataName,model, 'hmc', K);
        
        % Run HMC and get Results
        if ~exist([fileName,'.mat'],'file')
            exptEFLVM(dataName,model,K, seed);
        end;
        res = load(fileName);
        
        % Load Data
        dat = getBinData(dataName, dataDir);
        X = dat.X;
        [N,D] = size(dat.X);
        
        [S,LL] = size(res.postDist.samples);
        
        omega = res.postDist.samples(end,:);
        [V, Theta , Sigma, mu] = extractParams(omega,D, N, K);
        
        for i = 1:N
            if strfind(dat.lab{i},'(R-')
                RD(i) = 1;
            elseif strfind(dat.lab{i},'(D-')
                RD(i) = 2;
            elseif strfind(dat.lab{i},'(I-')
                RD(i) = 3;
            else
                RD(i) = 4;
            end;
        end;
        
        % Compute Marginal Covariance
        burnin = S;
        thin = 1;
        rng = burnin:thin:S;
        didx = 30;
        prob = eye(didx,didx);
        for s = rng
            s
            for i = 1:didx
                for j = 1:didx
                    omega = res.postDist.samples(s,:);
                    [V, Theta , Sigma, mu] = extractParams(omega,D, N, K);
                    pStar = mu'*Theta;
                    if i ~= j
                        prob(i,j) = sigmoid(pStar(i)).*sigmoid(pStar(j));
                        prob(i,j) = prob(i,j) + (1-sigmoid(pStar(i)))*(1-sigmoid(pStar(j)));
                    end;
                end;
            end;
        end;
        
        prob = prob./length(rng);
        % -- Generate plot
        % Visualise latent embedding
        figure;
        subplot(1,2,1);
        reps = RD==1;
        plot(V(reps,1), V(reps,2),'rs', 'LineWidth',2,'MarkerSize',14);
        hold on;
        dems = RD==2;
        plot(V(dems,1), V(dems,2),'bo', 'LineWidth',2,'MarkerSize',14);
        hold on
        oth = RD==3;
        plot(V(oth,1), V(oth,2),'gx', 'LineWidth',2,'MarkerSize',14);
        hold on;
        oth = RD==4;
        plot(V(oth,1), V(oth,2),'k*', 'LineWidth',2,'MarkerSize',14);
        title('Latent Factor Embedding','FontSize',14, 'FontWeight','bold');
        xlabel('Factor 1','FontSize',14, 'FontWeight','bold');
        ylabel('Factor 2','FontSize',14, 'FontWeight','bold');
        legend({'Republicans','Democrats','Independent','Outcome'});
        set(gca,'fontsize',14,'FontWeight','bold');
        axis square
        
        % Create PDFs and move to correct directory
        printPDF('efaVotes_embedding',outDir,print);
        
        figure;
        subplot(1,2,1)
        imagesc(prob(1:30,1:30));
        colorbar;
        title('Prob(x_i = x_j)','FontSize',14, 'FontWeight','bold');
        xlabel('Roll Call','FontSize',14, 'FontWeight','bold');
        ylabel('Roll Call','FontSize',14, 'FontWeight','bold');
        caxis([0 1]);
        hc = colorbar;
        set(hc, 'ytick', [0 0.2 0.4 0.6 0.8 1],'fontsize',11,'color',[.3 .3 .3]);
        set(gca,'fontsize',12,'FontWeight','bold');
        daspect([1 1 1]);
        
        % Show predictive probs
        pStar = sigmoid(V*Theta);
        X = dat.X;
        for i = 1:100
            hh = -((X ==1).*log2(pStar) + (X == 0).*log2(1-pStar));
        end;
        hh2 = sum(hh,2);
        disp('Pred Prob (nats): mean min median max')
        disp([mean(hh2) min(hh2) median(hh2) max(hh2)]);
        disp('Outcome (nats)');
        disp(hh2(end));
        
        % Create PDFs and move to correct directory
        printPDF('efaVotes_cov',outDir,print);
        
    case 'fuzzykm'
        %% 5. Senate votes using Fuzzy k-means
        dataName = 'votes';
        model = 'fuzzykm';
        K = 2;
        seed = 1;
        
        % Get directories
        [dataDir, outDir] = setupDir;
        fileName = sprintf([outDir '/%s/%s_%s_%d'], dataName,model, 'hmc', K);
        
        % Load Data
        dat = getBinData(dataName, dataDir);
        [N,D] = size(dat.X);
        X = dat.X';
        idx0 = X == 0;
        idxm = X == -1;
        X(idx0) = -1;
        X(idxm) = 0;
        
        maxiter=200;
        toldif=0.000001;
        %   distance type: 1 = euclidean, 2 = diagonal, 3 = mahalanobis
        distype=1; 
        scatter=0.2;
        ntry=1;
        
        res = [];
        coords = [83, 87, 100];
        res(:,1) = coords';
        rng = 1.1:0.1:3;
        for i = 1:length(rng)
            phi = rng(i);
            [U, centroid, dist, W, obj] = run_fuzme(K,X,phi,maxiter,distype,toldif,scatter,ntry);
            res(1,i+1) = max(U(coords(1),1), U(coords(1),2));
            res(2,i+1) = min(U(coords(2),1), U(coords(2),2));
            res(3,i+1) = min(U(coords(3),1), U(coords(3),2));
        end;
        
        col = 'brg';
        for i = 1:3
            plot(rng,res(i,2:end)','-s','LineWidth',2,'color',col(i));
            hold on;
        end;
        xlim([1 3.1]);
        ylim([-0.05 1.05]);
        ylabel('Partial Membership','FontSize',14, 'FontWeight','bold');
        xlabel('Fuzzy Exponent','FontSize',14, 'FontWeight','bold');
        set(gca,'fontsize',12,'FontWeight','bold');
        grid on
        legend(dat.lab(coords));
        
        % Create PDFs and move to correct directory
        printPDF('fuzzyVotes',outDir,print);
        
    otherwise
        error('no plotting routine');
end;




