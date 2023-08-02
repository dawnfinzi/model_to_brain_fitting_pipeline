function compare_model2subj_byarea(subjid,hemi,roi_name,area,mapping_func,CV,subsamp,subj_keys)
% compare_model2subj_by_area(subjid,hemi,roi_name,area,mapping_func,CV,subsamp)
%
%
% Plots requested model fits for a subject & area against the subj to subj
% mappings for the voxel
% subjid: (str) of form 'subj0X'
% hemi: (str) 'rh' or 'lh'
% roi_name: (str) full streams roi set to pull from
% area: (str) 
% mapping_func: (str) mapping used for the model fitting
% CV: (int) 0 = not cross-validated, 1 = cross-validated; refers to CV of
% params for model fitting function, all R^2 (regardless of CV) are on
% held-out test data
% subamp: (int) 0 = features were not subsampled, 1 = subsampled
% subj_keys: (cell) other subjects to pull from
%
% Default input values
% subjid          'subj02'
% hemi            'rh'
% roi_name        'streams_shrink10'
% area            'Ventral'
% mapping_func    'Ridge'
% CV              0
% subsamp         1
% subj_keys       {'01', '05', '07'}
% 
% DF 2021


%% Setup default inputs
if notDefined('subjid')
    subjid = 'subj02';
end
if notDefined('hemi')
    hemi = 'rh';
end
if notDefined('roi_name')
    roi_name = 'streams_shrink10';
end
if notDefined('area')
    area = 'Ventral';
end
if notDefined('mapping_func')
    mapping_func = 'Ridge';
end
if notDefined('CV')
    CV = 0;
end
if notDefined('subsamp')
    subsamp = 1;
end
if notDefined('subj_keys')
    subj_keys = {'01', '05', '07'};
end

%% Set other vars
fits_by_area_path = '/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/results/fits_by_area';
subj_fits_path = '/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/results/fits/subj2subj';
results_path = '/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/results/fits_by_area/figures';
layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7'};
rois = {'Early', 'Midventral', 'Midlateral', 'Midparietal', 'Ventral', 'Lateral', 'Parietal'};

a=find(strcmp(rois,area)==1); %index of area
subjix = str2num(subjid(end)); %index of subject
%% load subject to subject fits
subj_fits = struct([]);
all = [];
for s = 1:length(subj_keys)
   subj_fits(1).(['s' subj_keys{s}])= h5read(fullfile(subj_fits_path,['/', subjid, '_', hemi,'_', roi_name, '_', area, '_othersubjs_Ridge_fits.hdf5']), ['/',subj_keys{s}]);
   all = [all subj_fits(1).(['s' subj_keys{s}])];
end

all_m = mean(all, 2);

%% pull in fits by area
fits = struct([]);

for l = 1:length(layers)
    if subsamp == 0
        fits(1).(layers{l})= h5read(fullfile(fits_by_area_path,[subjid, '_', hemi,'_', roi_name, '_', area , '_alexnet_', mapping_func, num2str(CV), 'CV_fullfeats_fits.hdf5']), ['/',layers{l}]);
    else
        fits(1).(layers{l})= h5read(fullfile(fits_by_area_path,[subjid, '_', hemi,'_', roi_name, '_', area , '_alexnet_', mapping_func, num2str(CV), 'CV_fits.hdf5']), ['/',layers{l}]);
    end
end


num_vox = length(fits().(layers{1}));

for v = 1:num_vox
    rs_by_layer = [];
    for l = 1:length(layers)
        rs_by_layer = [rs_by_layer fits(1).(layers{l})(v)];
    end

    best_layer_by_vox(v) = find(rs_by_layer == max(rs_by_layer));
    max_rs(v) = max(rs_by_layer);   
end   


%% plot and save
cmap   = {[166/255, 166/255, 166/255]; [244/255, 189/255, 216/255];...
    [204/255, 218/255, 255/255]; [179/255, 255/255, 198/255]; [220/255, 38/255, 127/255];...
    [77/255, 127/255, 255/255]; [0/255, 102/255, 0/255]}; 
rColors = cmap;
fontSize = 11; titleSize = 14;
f(1) = niceFig([.1 .1 .4 .5],fontSize,2);

figure(f(1));

s = scatter(all_m,max_rs,.5,rColors{a}); hold on;
%s.MarkerEdgeAlpha = 0.25;
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = ':';

xl = xlim; xlim([0 1]); yl = ylim; ylim([0 1]);
set(gca,'TickDir','out');
ylabel('Model fits (R^2)','FontSize',fontSize); xlabel(['Subject to Subject (R^2)'],'FontSize',fontSize); 
axis square; 
title(area);


figure(f(1));
niceSave(results_path,sprintf('/%s_ModelFitsvsSubj2Subj_%s_%s_%s_%sCV_%ssubsamp', subjid, roi_name, area, mapping_func, num2str(CV), num2str(subsamp)),[],[],{'png' 'svg'});


% figure;hist(all_m-max_rs')
% mean(all_m-max_rs')
