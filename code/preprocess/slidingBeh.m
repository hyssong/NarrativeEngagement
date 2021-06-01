% Group-average behavioral timecourse generation for fMRI mapping
% HRF convolution and tapered sliding window 
% Code by Hayoung Song (hyssong@uchicago.edu)

% Input:  story = 'paranoia' or 'sherlock'

% Output: sliding-engagement.mat for multiple sliding window choices in path+'/data_processed' directory
%         (nT-wsize, 1)

function engagement_window = slidingBeh(story)

path = fileparts(fileparts(pwd)); % 2 steps parent directory

% Loads engagement time course and hyperparameters
load([path,'/data/behavior-engagement-',story,'.mat']);
load([path,'/data/hyperparameters.mat'],[story,'_wsize']);
load([path,'/data/hyperparameters.mat'],[story,'_sigma']);
load([path,'/data/hyperparameters.mat'],[story,'_TR']);

nT = size(engagement,1);
wsize = eval([story,'_wsize']);
sigma = eval([story,'_sigma']);
TR = eval([story,'_TR']);

disp(' ');
disp('Apply sliding-window to behavioral timecourse');
disp([' story         : ',story]);
disp([' ntime         : ',num2str(nT)]);
disp([' TR            : ',num2str(TR),' s']);
disp([' wsize         : [ ',num2str(wsize),' ]']);
disp([' tapered sigma : ',num2str(sigma)]);
disp(' step size     : 1 TR');
disp(' ');

engagement_window = {};
for ws = 1:length(wsize)
    disp(['    wsize         = ',num2str(wsize(ws))]);
    
    % normalize each individual's continuous behavioral ratings, then average
    engagement_z = [];
    for subj = 1:size(engagement,2)
        engagement_z = [engagement_z, zscore(engagement(:,subj))];
    end
    engagement = mean(engagement_z,2);
    
    % hemodynamic response function convolution to later map with fMRI timecourse
    weight = [0,0.000354107958396228,0.0220818694830938,0.116001537001027,0.221299059999514,0.242353095826523,0.186831750619196,0.113041009515928,0.0572809597709863,0.0253492394574127,0.0100814114758446,0.00367740475539297,0.00124901102357508,0.000399543113110135,0];
    if mod(TR,1)==0
        weight = resample(weight,1,TR);
    else
        trratio = strsplit(rats(TR),'/');
        weight = resample(weight,str2num(trratio{2}),str2num(trratio{1}));
    end

    conv_engagement = conv(engagement,weight');
    conv_engagement = conv_engagement(1:nT);
    
    % sliding window apply: see function below
    sliding_engagement = slidingwindow(conv_engagement, nT, wsize(ws), sigma);
    
    % save preprocessed data
    if exist([path,'/data_processed/',story,'/win',num2str(wsize(ws))])==0
        mkdir([path,'/data_processed/',story,'/win',num2str(wsize(ws))]);
    end
    save([path,'/data_processed/',story,'/win',num2str(wsize(ws)),'/sliding-engagement.mat'],'sliding_engagement');
    engagement_window{ws,1} = sliding_engagement;
end
end


function sliding_ts = slidingwindow(ts, nT, wsize, sigma)
% code created by Bo-yong Park: https://by9433.wixsite.com/boyongpark
if size(ts,1)~=nT
    error('number of rows of variable "ts" should match a variable "nT".');
end

% tapered sliding window: generate gaussian function for convolution
% Allen et al. 2014, Cerebral Cortex
if mod(nT,2) ~= 0
    m = ceil(nT/2);
    x = 0:nT;
else
    m = nT/2;
    x = 0:nT-1;
end

w = round(wsize/2);
gw = exp(- ((x-m).^2) / (2*sigma*sigma))';
b = zeros(nT,1); b((m-w+1):(m+w)) = 1;
c = conv(gw, b); c = c/max(c); c = c(m+1:end-m+1);
c = c(1:nT);

A = repmat(c,1,1);
Nwin = nT - wsize;
FNCdyn = zeros(Nwin, 1);

% apply sliding window
tcwin = zeros(Nwin, nT);
for ii = 1:Nwin
    % slide gaussian centered on [1+wsize/2, nT-wsize/2]
    Ashift = circshift(A, round(-nT/2) + round(wsize/2) + ii);
    
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
    tcwin(ii, :) = squeeze(ts).*Ashift;
end

% normalize for a final round after sliding-window
sliding_ts = zscore(nansum(tcwin,2));
end