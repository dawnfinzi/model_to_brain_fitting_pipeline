%% Compare model fits to subject2subject fits

close all 
clear all

oak_stem = '/oak/stanford/groups/kalanit/biac2/kgs/projects/';
share_stem =  '/share/kalanit/biac2/kgs/projects/';

local_data_path = fullfile(oak_stem, 'Dawn/NSD/local_data');
fits_path = fullfile(oak_stem, 'Dawn/NSD/results/fits');
results_path = fullfile(oak_stem, 'Dawn/NSD/results/fits/figures');

roi_name = 'streams';
subj2subj = 0;
NC_type = 1; %0 = split-half, 1 = Kendrick's NC metric
method = 'PLS';
CV = 0;

layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7'};
rois = {'Early', 'MidVent', 'MidLat', 'MidPar', 'Ventral', 'Lateral', 'Parietal'};

hemis = {'lh', 'rh'}; 
subj_keys = {'01', '02', '03', '04', '05', '06', '07', '08'};

byroi = struct([]);
for s = 1:length(subj_keys)
    subjid = ['subj' subj_keys{s}];  
    subjix=s;

    %% load noise ceiling maps 
    full_nc = struct([]);
    if NC_type == 0
        %split-half estimate
        for h = 1:length(hemis)
            nc = load(sprintf('/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/local_data/freesurfer/%s/%s_split_half.mat', subjid,hemis{h}));
            full_nc(1).(hemis{h}) = nc.mean';
        end
    else
        % NC3 estimates
        data_dir = sprintf('%s/ppdata/subj%02d/nativesurface/betas_fithrf_GLMdenoise_RR/',nsd_datalocation('betas'),subjix);  
        for h = 1:length(hemis)
            nc = load_mgh([data_dir  sprintf('%s.nc_3trials.mgh',hemis{h})]); load(sprintf('/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/local_data/freesurfer/%s/%s_split_half.mat', subjid,hemis{h}));
            full_nc(1).(hemis{h}) = nc';
        end
    end

    %% load model fits
    fits = struct([]);
    for h = 1:length(hemis)
        for l = 1:length(layers)
            fits(h).(layers{l})= h5read(fullfile(fits_path,[subjid, '_', hemis{h},'_', roi_name, '_alexnet_', method, num2str(CV), 'CV_fits.hdf5']), ['/',layers{l}]);
        end
    end

    %% calculate best layer for each voxel
    metrics = struct([]);
    for h = 1:length(hemis)

        num_vox = length(fits(h).(layers{1}));

        for v = 1:num_vox
            rs_by_layer = [];
            for l = 1:length(layers)
                rs_by_layer = [rs_by_layer fits(h).(layers{l})(v)];
            end

            metrics(h).best_layer_by_vox(v) = find(rs_by_layer == max(rs_by_layer));
            metrics(h).max_rs(v) = max(rs_by_layer);   
        end   
    end

    %% format to work with full flat map using streams rois
    idx = struct([]);
    best_layer = []; best_rs = []; 
    join = [];
    nc = [];
    for h = 1:length(hemis)
        roivals = cvnloadmgz(sprintf('/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/data/nsddata/freesurfer/%s/label/%s.%s.mgz',subjid, hemis{h}, roi_name));  % load in an existing file?
        join = [join; roivals];

        idx(1).(hemis{h}) = find(roivals ~= 0);

        temp = zeros(length(full_nc(1).(hemis{h})),1)';
        temp(idx(1).(hemis{h})) = metrics(h).best_layer_by_vox;
        best_layer = [best_layer temp];

        temp = zeros(length(full_nc(1).(hemis{h})),1)';
        temp(idx(1).(hemis{h})) = metrics(h).max_rs;
        best_rs = [best_rs temp];

        nc = [nc full_nc(1).(hemis{h})];
    end
    %% order results by ROI

    for r = 1:length(rois)
        if s == 1 %first subject
            byroi(r).best_layer = best_layer(join == r);
            byroi(r).best_rs = best_rs(join == r);
            byroi(r).nc = nc(join == r);
        else
            byroi(r).best_layer = [byroi(r).best_layer best_layer(join == r)];
            byroi(r).best_rs = [byroi(r).best_rs best_rs(join == r)];
            byroi(r).nc = [byroi(r).nc nc(join == r)];
        end
    end
end

%% test nice subplot
cmap   = {[166/255, 166/255, 166/255]; [244/255, 189/255, 216/255];...
    [204/255, 218/255, 255/255]; [179/255, 255/255, 198/255]; [220/255, 38/255, 127/255];...
    [77/255, 127/255, 255/255]; [0/255, 102/255, 0/255]}; 
rColors = cmap;

ROIs = rois;
ridx = [1,2,3,4,5, 6, 7];
fontSize = 11; titleSize = 14;
f(1) = niceFig([0 0 .9 .3],fontSize,2);

for r = 1:length(ROIs)
    figure(f(1));
    subplot(1,7,r);
    
    %s = scatter(x,byroi(ridx(r)).best_rs,2,rColors{r}); hold on;
    s = scatter(byroi(ridx(r)).nc./100,byroi(ridx(r)).best_rs,1,rColors{r}); hold on;
    %s = scatter(x, byroi(ridx(r)).nc./100,2,rColors{r}); hold on;
    %s.MarkerEdgeAlpha = 0.25;
    hline = refline([1 0]);
    hline.Color = 'k';
    hline.LineStyle = ':';
    
    xl = xlim; xlim([0 1]); yl = ylim; ylim([0 1]);
    set(gca,'TickDir','out');
    ylabel('Model fits (R^2)','FontSize',fontSize); xlabel(['NC (R^2)'],'FontSize',fontSize); 

    %ylabel('Model fits (R^2)','FontSize',fontSize); xlabel(['Split Half (r)'],'FontSize',fontSize); 
    %xlabel('Subj fits (R^2)','FontSize',fontSize); ylabel(['NC (R^2)'],'FontSize',fontSize); 
    %xlabel('Other subject fits (R^2)','FontSize',fontSize); ylabel(['Model fits (R^2)'],'FontSize',fontSize); 
    axis square; 
    title(ROIs{r});
end

figure(f(1));
niceSave(results_path,sprintf('/allsubjs_ModelFitsvsNC3_allROIs_%s', roi_name),[],[],{'png'});

%% Higher visual areas

rColors = {[220/255, 38/255, 127/255];...
    [77/255, 127/255, 255/255]; [0/255, 102/255, 0/255]
    };

ROIs = {'Ventral', 'Lateral', 'Parietal'};
ridx = [5, 6, 7];

%% NC vs model
fontSize = 18; titleSize = 28;
f(1) = niceFig([.1 .1 1.2 .6],fontSize,2);
line_means = [];
for r = 1:length(ROIs)
    figure(f(1));
    subplot(1,3,r);
    
    if NC_type == 0
        NC = byroi(ridx(r)).nc;
    else
        NC = byroi(ridx(r)).nc./100; %divide by 100
    end
    
    fitRange = [0, .05, 1];
    sampleVox =40000;
    sv = datasample([1:length(NC)], sampleVox, 'Replace', false);
    
    s = scatter(NC(sv),byroi(ridx(r)).best_rs(sv),.1,rColors{r},'o'); hold on;
    %s.MarkerEdgeAlpha = 0.25;
    hline = refline([1 0]);
    hline.Color = 'k';
    hline.LineStyle = ':';
    
    [errorObj, lineObj, m, s] = scatterline(NC,byroi(ridx(r)).best_rs,fitRange,NaN,100,[0,0,0],0,1); hold on;
    set(lineObj, 'LineWidth', 1.5);
    alpha(errorObj,.75)
    
    xl = xlim; xlim([0 1]); yl = ylim; ylim([0 1]);
    set(gca,'TickDir','out');
    if NC_type == 1
        ylabel('Model fits (R^2)','FontSize',fontSize); xlabel(['NC (R^2)'],'FontSize',fontSize); 
    else
        ylabel('Model fits (R^2)','FontSize',fontSize); xlabel(['Split Half (r)'],'FontSize',fontSize);
    end
    axis square;
    title(ROIs{r},'FontSize',fontSize);
    
    line_means(r,:) = m;
end

figure(f(1));
if NC_type == 1
    niceSave(results_path,sprintf('/allsubjs_ModelFitsvsNC3_%s', roi_name),[],[],{'png'});
else
   niceSave(results_path,sprintf('/allsubjs_ModelFitsvsSplitHalf_%s', roi_name),[],[],{'png'});
end 
