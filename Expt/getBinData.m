function data = getBinData(dataName, dataDir)

setSeed(1);
switch dataName
    case 'bpmmodel'
        data = genpartmembin(50,[5 5 5],0.74);
        
    case 'votes'
        disp('US Senate Voting Data');
        fname = sprintf('%s/%s',dataDir,'SenRC1');
        sen = load(fname);
        X = sen.SD; % 100 senators x 633 bills
        
        data.X = X;
        data.trueX = X;
        data.lab = sen.SL;
        
    case 'synth'
        pat = rand(3,16)> 0.5;
        rep = 100;
        Xclean = [];
        for i = 1:size(pat,1)
            Xclean = [Xclean; repmat(pat(i,:),rep,1)];
        end;
        
        [R,C] = size(Xclean);
        Xfinal = Xclean;
        for r = 1:R
            for c = 1:C
                if rand<0.05
                    Xfinal(r,c) = 1-Xclean(r,c);
                end
            end;
        end;
        Y = Xfinal; % Use noisy data
        
        % Create missing/test data
        ns = numel(Y);
        nmiss = ceil(0.1*ns); % percentage of data as missing/test
        idx = randperm(ns);
        Ymiss = Y;
        Ymiss(idx(1:nmiss)) = -1;
        
        % obs data is 0/1 and missing is -1.
        data.X = Ymiss;
        data.trueX = Y;
        data.cleanX = Xclean;
        data.miss = idx(1:nmiss);
        
        
    case 'blockImages'
        % create binary data: 100 x 36
        dim_count = 36;
        data_count = 100;
        Z = double( rand( data_count , 4 ) > .66 );
        A = zeros( 4 , dim_count );
        f1idx = [ 2 7 8 9 14 ]; A( 1 , f1idx ) = 1;
        f2idx = [ 4 5 6  10 12 16 17 18 ]; A( 2 , f2idx ) = 1;
        f3idx = [ 19 25 26 31 32 33 ]; A( 3 , f3idx ) = 1;
        f4idx = [ 22 23 24 29 35 ]; A( 4 , f4idx ) = 1;
        Xclean = Z * A;
        
        [R,C] = size(Xclean);
        Xfinal = Xclean;
        for r = 1:R
            for c = 1:C
                if rand<0.05
                    Xfinal(r,c) = 1-Xclean(r,c);
                end
            end;
        end;
        Y = Xfinal'; % Use noisy data
        
        % Create test data
        missRatio = 0.5;
        Ymiss = Y;
        [D,N] = size(Y);
        d = unidrnd(D,1,N); % one missing dim in each data point
        nMiss = 0;
        for i = 1:N
            if rand < missRatio
                Ymiss(d(:,i),i) = -1;
                nMiss = nMiss + 1; % #miss
            end;
        end;
        miss = find(Ymiss==-1);
        
        % obs data is 0/1 and missing is -1.
        data.X = Ymiss;
        data.miss = miss;
        data.trueX = Y;
        data.cleanX = Xclean;
        
    case 'led'
        dat = load('uci.led17.mat'); % 2000x24
        
        % Create test data
        missRatio = 0.5;
        Ymiss = dat.X';
        [D,N] = size(Ymiss);
        d = unidrnd(D,1,N); % one missing dim in each data point
        nMiss = 0;
        for i = 1:N
            if rand < missRatio
                Ymiss(d(:,i),i) = -1; % missing is -1 for epca
                nMiss = nMiss + 1; % #miss
            end;
        end;
        miss = find(Ymiss==0);
        
        % return formatted data
        data.trueX = dat.X';
        data.miss = miss;
        data.X = Ymiss;
        
    case {'german', 'australian', 'pima', 'ripley', 'heart'}
        % Get data
        fname = sprintf('%s%s.mat',dataDir,dataName);
        in = load(fname);
        
        % Shape data
        tmp = in.X(:,1:end-1); % DxN
        Y = in.X(:,end); % Nx1;
        Xorig = zscore(tmp)';
        
        switch dataName
            case 'german'
                nTestPerClass = 50;
                polyOrder = 1;
                Y(Y == 2) = 0; % switch to 1/0 convention
            case 'australian'
                nTestPerClass = 35;
                polyOrder = 1;
            case 'pima'
                nTestPerClass = 25;
                polyOrder = 1;
            case 'ripley'
                nTestPerClass = 15;
                polyOrder = 1;
            case 'heart'
                nTestPerClass = 13;
                polyOrder = 1;
                Y(Y == 2) = 0; % switch to 1/0 convention
        end;
        
        % Create data for poly basis regression if desired
        X = [];
        for i = 1:polyOrder
            X = [X; Xorig.^i];
        end;
        
        % Create Train and Test data
        ix1 = find(Y == 1);
        testLoc1 = randperm(length(ix1));
        ix0 = find(Y == 0);
        testLoc0 = randperm(length(ix0));
        
        testIx = [ix0(testLoc0(1:nTestPerClass)); ix1(testLoc1(1:nTestPerClass))];
        data.Xtest = X(:,testIx)';
        data.Ytest = Y(testIx);
        
        trainIx = [ix0(testLoc0(nTestPerClass+1:end)); ix1(testLoc1(nTestPerClass+1:end))];
        data.X = X(:,trainIx)';
        data.Y = Y(trainIx);
        
    otherwise
        error('no such data name');
end;