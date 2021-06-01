% Time-resolved functional connectivity matrix generation
% Code by Hayoung Song (hyssong@uchicago.edu)

% The code loads BOLD timeseries of all participants (nsubj, ROI, nT_subj)
% The code outputs tapered sliding window applied dynamic functional connectivity

% Input:  story = 'paranoia' or 'sherlock'

% Output: fmri-dynFeat-story.mat at path+'/data_processed' directory
%         (nsubj, nRx(nR-1)/2, nT-wsize)

function dynFeat_win = slidingFC(story)

path = fileparts(fileparts(pwd)); % 2 steps parent directory

% Loads BOLD timecourse and hyperparameters
load([path,'/data/fmri-BOLD-',story,'.mat']);
load([path,'/data/hyperparameters.mat'],[story,'_wsize']);
load([path,'/data/hyperparameters.mat'],[story,'_sigma']);

wsize = eval([story,'_wsize']);
sigma = eval([story,'_sigma']);
nsubj = size(BOLD,1);
nR = size(BOLD,2);
nT = size(BOLD,3);

disp(' ');
disp('Create dynamic functional connectivity');
disp(['  story         = ',story]);
disp(['  nsubj         = ',num2str(nsubj)]);
disp(['  nregion       = ',num2str(nR)]);
disp(['  ntime         = ',num2str(nT)]);

dynFeat_win = {};
for ws = 1:length(wsize)
    dynFeat = [];
    disp(' ');
    disp(['    wsize         = ',num2str(wsize(ws))]);
    disp(['    stepsize      = 1']);
    disp(['    tapered sigma = ',num2str(sigma)]);
    disp(' ');
    for subj = 1:nsubj
        disp(['    subj ',num2str(subj),' / ',num2str(nsubj)]);
        ts = squeeze(BOLD(subj,:,:)); ts=ts';
        
        if any(isnan(ts(:,1)))
            nT_subj = length(find(~isnan(ts(:,1))));
            nanid = find(isnan(ts(:,1)));
            disp('******* include missing values in the timeseries');
            ts = ts(find(~isnan(ts(:,1))),:);
        else
            nT_subj = nT;
        end
        
        % code created by Bo-yong Park: https://by9433.wixsite.com/boyongpark
        % Normalize within region, then divide it by the total stddev
        grot=ts;
        grot=grot-repmat(mean(grot),size(grot,1),1); % demean
        grot=grot/std(grot(:)); % normalise whole subject stddev
        ts = grot;
        
        % compute sliding window
        if size(ts,1)~= nT_subj
            error('check time series column/row');
        end
        if mod(nT_subj,2) ~= 0
            m = ceil(nT_subj/2);
            x = 0:nT_subj;
        else
            m = nT_subj/2;
            x = 0:nT_subj-1;
        end
        w = round(wsize(ws)/2);
        gw = exp(- ((x-m).^2) / (2*sigma*sigma))';
        b = zeros(nT_subj,1); b((m-w+1):(m+w)) = 1;
        c = conv(gw, b); c = c/max(c); c = c(m+1:end-m+1);
        c = c(1:nT_subj);
        
        % Dynamic connectivity
        A = repmat(c,1,nR);
        Nwin = nT_subj - wsize(ws);
        FNCdyn = zeros(Nwin, nR*(nR - 1)/2);
        
        % Apply circular shift to time series
        tcwin = zeros(Nwin, nT_subj, nR);
        for ii = 1:Nwin
            % slide gaussian centered on [1+wsize/2, nT_subj-wsize/2]
            Ashift = circshift(A, round(-nT_subj/2) + round(wsize(ws)/2) + ii);
            
            % when using "circshift", prevent spillover of the gaussian
            % to either the beginning or an end of the timeseries
            if ii<floor(Nwin/2) & Ashift(end,1)~=0
                Ashift(ceil(Nwin/2):end,:) = 0;
                Ashift = Ashift.*(sum(A(:,1))/sum(Ashift(1:floor(Nwin/2),1)));
            elseif ii>floor(Nwin/2) & Ashift(1,1)~=0
                Ashift(1:floor(Nwin/2),:) = 0;
                Ashift = Ashift.*(sum(A(:,1))/sum(Ashift(ceil(Nwin/2):end,1)));
            end
            
            % apply gaussian weighted sliding window of the timeseries
            tcwin(ii, :, :) = squeeze(ts).*Ashift;
        end
        
        % Fisher's r-to-z transformed dynamic functional connectivity matrix
        tapered_pearsonz = zeros(Nwin,nR,nR);
        for ii = 1:Nwin
            tmp = atanh(corr(squeeze(tcwin(ii,:,:))));
            for i = 1:nR; tmp(i,i) = 0; end
            tapered_pearsonz(ii,:,:) = tmp;
        end
        
        if nT_subj~=nT
            disp('******* add NaN dynFC at the end');
            tapered_pearsonz = cat(1,tapered_pearsonz,zeros(length(nanid),nR,nR)*NaN);
        end
        
        % to reduce data size, reduce into feature dimension
        dynft = [];
        for tm = 1:(nT-wsize(ws))
            tmp = squeeze(tapered_pearsonz(tm,:,:));
            feat = [];
            for i1 = 1:nR-1
                for i2 = i1+1:nR
                    feat = [feat; tmp(i1,i2)];
                end
            end
            dynft = [dynft, feat];
        end
        dynFeat = cat(3, dynFeat, dynft);
    end
    
    % dynamic brain connectivity "feature": (nsubj, nRx(nR-1)/2, nT-wsize)
    dynFeat = permute(dynFeat, [3,1,2]);
    
    disp('Saving .....');
    if exist([path,'/data_processed/',story,'/win',num2str(wsize(ws))])==0
        mkdir([path,'/data_processed/',story,'/win',num2str(wsize(ws))]);
    end
    save([path,'/data_processed/',story,'/win',num2str(wsize(ws)),'/sliding-dynFeat.mat'],'dynFeat');
    disp('Finished!');
    
    dynFeat_win{ws,1} = dynFeat;
end
end
