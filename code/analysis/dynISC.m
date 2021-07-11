% Dynamic inter-subject correlation, correlated with behavior
% Code by Hayoung Song (hyssong@uchicago.edu)

% Generate pairwise inter-subject correlation (ISC), calculated across time using sliding window
% Per ROI, correlate mean dynamic ISC with group-level behavior
% Outputs statistics (two-tailed non-parametric permutation test)
% (optional) Projects dynamic ISC - behavior correlation to MNI space for visualization

% Input:  story = 'paranoia' or 'sherlock;
%         wsize = one of [32,40,48] for 'paranoia', one of [24,30,36] for 'sherlock'
%         thres = 0.05; % uncorrected significance threshold for
%         non-parametric permutation statistical test

% Output: ISC.mat
%         - results: significant ROI index, Pearson's r with behavior, uncorrected p
%         - dynISC: (nsubj x (nsubj-1) / 2, nT-wsize, nROI)
% Output (optional): ISC.nii


function [results, dynISC] = dynISC(story, wsize, thres)
path = fileparts(fileparts(pwd)); % 2 steps parent directory

if nargin < 2
    if strcmp(story,'paranoia')
        wsize = 40;
    elseif strcmp(story,'sherlock')
        wsize = 30;
    end
    thres = 0.05;
elseif nargin < 3
    thres = 0.05; % default 0.05
end

% load BOLD timecourse of all participants
load([path,'/data/fmri-BOLD-',story,'.mat']);
nsubj = size(BOLD,1);
nT = size(BOLD,3);
nR = size(BOLD,2);

% Loads hyperparameters
load([path,'/data/hyperparameters.mat'],[story,'_sigma']);
sigma = eval([story,'_sigma']);

disp(['story                = ',story]);
disp(['window size          = ',num2str(wsize)]);
disp(['tapered window sigma = ',num2str(sigma)]);
disp(['nsubj                = ',num2str(nsubj)]);
disp(['ntime                = ',num2str(nT)]);
disp(['nregion              = ',num2str(nR)]);
disp(['p-threshold          = ',num2str(thres)]);

savepath = [path,'/result/dynISC/',story,'/win',num2str(wsize)];
if exist(savepath)==0
    mkdir(savepath);
end

%% Load all subjects' sliding windowed time series
% tcwin: (nT-wsize) x nT x nR
% data saved per participant
disp(' ');
disp('Create sliding window time series');
for subj = 1:nsubj
    disp(['  subj ',num2str(subj),' / ',num2str(nsubj)]);
    ts = squeeze(BOLD(subj,:,:)); ts = ts';
    
    if any(isnan(ts(:,1)))
        nT_subj = length(find(~isnan(ts(:,1))));
        nanid = find(isnan(ts(:,1)));
        disp('  ******* due to fMRI timeseries having NaN, temporarily erase NaN from timeseries');
        ts = ts(find(~isnan(ts(:,1))),:);
    end
    
    % Normalize within region, then divide it by the total stddev
    grot=ts;
    grot=grot-repmat(mean(grot),size(grot,1),1); % demean
    grot=grot/std(grot(:)); % normalise whole subject stddev
    ts = grot;
    
    % compute sliding window
    nT_subj = size(ts,1);
    nR = size(ts,2);
    
    if mod(nT_subj,2) ~= 0
        m = ceil(nT_subj/2);
        x = 0:nT_subj;
    else
        m = nT_subj/2;
        x = 0:nT_subj-1;
    end
    w = round(wsize/2);
    gw = exp(- ((x-m).^2) / (2*sigma*sigma))';
    b = zeros(nT_subj,1); b((m-w+1):(m+w)) = 1;
    c = conv(gw, b); c = c/max(c); c = c(m+1:end-m+1);
    c = c(1:nT_subj);
    
    % Dynamic connectivity
    A = repmat(c,1,nR);
    Nwin = nT_subj - wsize;
    FNCdyn = zeros(Nwin, nR*(nR - 1)/2);
    
    % Apply circular shift to time series
    tcwin = zeros(Nwin, nT_subj, nR);
    for ii = 1:Nwin
        % slide gaussian centered on [1+wsize/2, nT_subj-wsize/2]
        Ashift = circshift(A, round(-nT_subj/2) + round(wsize/2) + ii);
        
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
    
    if nT_subj~=nT
        disp('  ******* add NaN dynFC at the end');
        tcwin = cat(1, tcwin, zeros(nT-nT_subj, size(tcwin,2), size(tcwin,3))*NaN);
        tcwin = cat(2, tcwin, zeros(size(tcwin,1), nT-nT_subj, size(tcwin,3))*NaN);
    end
    
    save([savepath,'/tcwin',num2str(subj),'.mat'],'tcwin','-v7.3');
end

%% Pairwise
% sliding window applied dynamic ISC: (nT-wsize) x ROI
% data saved per participant pair
disp(' ');
disp('Pairwise dynamic ISC calculation');
idx = 0;
for subj1 = 1:nsubj-1
    load([savepath,'/tcwin',num2str(subj1),'.mat']);
    subj1_data = tcwin;
    for subj2 = subj1+1:nsubj
        disp(['  subj ',num2str(subj1),' & subj ',num2str(subj2)]);
        idx = idx+1;
        if exist([savepath,'/pair',num2str(idx),'.mat'])~=0
        else
            load([savepath,'/tcwin',num2str(subj2),'.mat']);
            subj2_data = tcwin;
            
            time_region_slide = [];
            for time = 1:nT-wsize
                subj1_tmp = squeeze(subj1_data(time,:,:));
                subj2_tmp = squeeze(subj2_data(time,:,:));
                region_slide = [];
                for region = 1:nR
                    region_slide = [region_slide; atanh(corr(subj1_tmp(:,region),subj2_tmp(:,region),'rows','complete'))];
                end
                time_region_slide = [time_region_slide, region_slide];
            end
            time_region_slide = time_region_slide';
            save([savepath,'/pair',num2str(idx),'.mat'],'time_region_slide');
        end
    end
end

%% DynISC - behavior correlation
% Load every pairwise dynamic ISC
% nPair x (nT-wsize) x nR
flist = dir([savepath,'/pair*.mat']);
pairwisesubj_corr = [];
dynISC = zeros(nsubj*(nsubj-1)/2, nT-wsize, nR);
for listf = 1:length(flist)
    load([savepath,'/pair',num2str(listf),'.mat']);
    dynISC(listf, :,:) = time_region_slide;
end

% Load sliding-window applied behavioral data
load([path,'/data_processed/',story,'/win',num2str(wsize),'/sliding-engagement.mat']);

% Per ROI, average pairwise participants' dynamic ISC
mean_dynISC = tanh(squeeze(nanmean(dynISC,1)));

% Correlate with behavioral data, per ROI
region_corr = [];
for region = 1:nR
    region_corr = [region_corr; corr(mean_dynISC(:,region),sliding_engagement,'rows','complete')];
end
region_corr_actual = region_corr;

% Comparison with null behavior
load([path,'/data_processed/',story,'/win',num2str(wsize),'/sliding-engagement-surr.mat']);
nsurr = size(sliding_surr_engagement,2);
disp(['permutation iteration = ',num2str(nsurr)]);
region_corr_surr = zeros(nR,nsurr);
for surr = 1:nsurr
    if mod(surr,100)==0
        disp(['surr ',num2str(surr),' / ',num2str(nsurr)]);
    end
    region_corr = [];
    for region = 1:nR
        region_corr = [region_corr; corr(mean_dynISC(:,region), sliding_surr_engagement(:,surr),'rows','complete')];
    end
    region_corr_surr(:,surr) = region_corr;
end

%% Statistics
% non-parametric permutation test (two-tailed)
twotailed_pval = [];
for region = 1:nR
    actual = region_corr_actual(region,1);
    if isnan(actual) == 1
        twotailed_pval = [twotailed_pval; NaN];
    else
        surrogate = region_corr_surr(region,:);
        pv = (1+length(find(abs(surrogate)>=abs(actual))))/(length(surrogate)+1);
        twotailed_pval = [twotailed_pval; pv];
    end
end

%% Save
% ROI index, dynamic ISC & behavior correlation r value, p value
results = [];
for roi = 1:nR
    if twotailed_pval(roi) < thres
        results = [results; roi, region_corr_actual(roi), twotailed_pval(roi)];
    end
end
save([savepath,'/ISC.mat'],'results','dynISC');

%% Load mask nifti file and project r values in the brain
% -------------------------- %
% You can run this code after creating whole-brain parcellation mask and saving the nifti file at [path, '/network']
% This projects significant correlation values (dynamic ISC - behavior) to the ROIs (MNI space)

% Yeo et al. (2015; Neuroimage) 114 cortical ROIs
% URL: https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/
% Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/MNI152
% Each cortical ROI corresponds to one of 7-networks (Yeo et al., 2011; J. Neurophysiol.)

% Brainnetome atlas 8 subcortical ROIs
% amygdala, hippocampus, basal ganglia, thalamus (LH and RH)
% URL: https://atlas.brainnetome.org
% -------------------------- %

% if strcmp(story,'paranoia')
%     masknii = load_nii([path,'/network/Yeo122_2mm.nii']);
% elseif strcmp(story,'sherlock')
%     masknii = load_nii([path,'/network/Yeo122_3mm.nii']);
% end
% newimg = zeros(size(masknii.img));
% oldimg = masknii.img;
% for i1 = 1:size(newimg,1)
%     for i2 = 1:size(newimg,2)
%         for i3 = 1:size(newimg,3)
%             if oldimg(i1,i2,i3) > 0
%                 tmp = oldimg(i1,i2,i3);
%                 if ismember(tmp,results(:,1))
%                     newimg(i1,i2,i3) = results(find(results(:,1)==tmp),2);
%                 end
%             end
%         end
%     end
% end
% masknii.img = newimg;
% save_nii(masknii,[savepath,'/ISC.nii']);

% cd(savepath);
% feel free to remove interim files: pair*.mat, tcwin*.mat
end