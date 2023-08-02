function compare_model2NC_byarea(subjid,hemi,roi_name,area,mapping_func,CV,subsamp)
% compare_model2NC_by_area(subjid,hemi,roi_name,area,mapping_func,CV,subsamp
%
%
% Plots requested model fits for a subject & area against the noise ceiling
% estimates by voxel
% subjid: (str) of form 'subj0X'
% hemi: (str) 'rh' or 'lh'
% roi_name: (str) full streams roi set to pull from
% area: (str) 
% mapping_func: (str) mapping used for the model fitting
% CV: (int) 0 = not cross-validated, 1 = cross-validated; refers to CV of
% params for model fitting function, all R^2 (regardless of CV) are on
% held-out test data
% subamp: (int) 0 = features were not subsampled, 1 = subsampled
%
% Default input values
% subjid          'subj02'
% hemi            'rh'
% roi_name        'streams_shrink10'
% area            'Ventral'
% mapping_func    'Ridge'
% CV              0
% subsamp         0
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
    subsamp = 0;
end

%% Set other vars
fits_by_area_path = '/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/results/fits_by_area';
results_path = '/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/results/fits_by_area/figures';
layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7'};
rois = {'Early', 'Midventral', 'Midlateral', 'Midparietal', 'Ventral', 'Lateral', 'Parietal'};

a=find(strcmp(rois,area)==1); %index of area
subjix = str2num(subjid(end)); %index of subject
%% get NC and sort by area
full_nc = struct([]);

% NC3 estimates
data_dir = sprintf('%s/ppdata/subj%02d/nativesurface/betas_fithrf_GLMdenoise_RR/',nsd_datalocation('betas'),subjix);  
nc = load_mgh([data_dir  sprintf('%s.nc_3trials.mgh',hemi)]); load(sprintf('/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/local_data/freesurfer/%s/%s_split_half.mat', subjid,hemi));
roivals = cvnloadmgz(sprintf('/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/data/nsddata/freesurfer/%s/label/%s.%s.mgz',subjid, hemi, roi_name));  % load in an existing file?

byroi = struct([]);
for r = 1:length(rois)
    
    byroi(r).nc = nc(roivals == r);
end

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

s = scatter(byroi(a).nc./100,max_rs,1,rColors{a}); hold on;
%s.MarkerEdgeAlpha = 0.25;
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = ':';

xl = xlim; xlim([0 1]); yl = ylim; ylim([0 1]);
set(gca,'TickDir','out');
ylabel('Model fits (R^2)','FontSize',fontSize); xlabel(['NC (R^2)'],'FontSize',fontSize); 
axis square; 
title(area);


figure(f(1));
niceSave(results_path,sprintf('/%s_ModelFitsvsNC_%s_%s_%s_%s_%sCV_%ssubsamp', subjid, hemi, roi_name, area, mapping_func, num2str(CV), num2str(subsamp)),[],[],{'png' 'svg'});
