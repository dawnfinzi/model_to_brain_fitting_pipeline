%% Load in model fitting results and plot on cortical surface

close all 
clear all

oak_stem = '/oak/stanford/groups/kalanit/biac2/kgs/projects/';
share_stem =  '/share/kalanit/biac2/kgs/projects/';

local_data_path = fullfile(oak_stem, 'Dawn/NSD/local_data');
fits_path = fullfile(oak_stem, 'Dawn/NSD/results/fits');
s2s_path = fullfile(oak_stem, 'Dawn/NSD/results/fits/subj2subj');
results_path = fullfile(oak_stem, 'Dawn/NSD/results/fits/figures');


hemis = {'rh'}; %'lh', 
subjid = 'subj08';  
subjix = 8;
model_name = 'slowfast_full1'; % 'resnet18';
roi_name = 'streams_shrink10';
n_comps = 25;
subsample = 2;
method = 'PLS';
CV = 0;

if strcomp(model_name, 'alexnet')
    layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7'};
    num_layers = 7;
elseif strcomp(model_name, 'alexnet_torch')
    layers = {'features.2', 'features.5', 'features.7', 'features.9', ...
              'features.12', 'classifier.2', 'classifier.5'};
    num_layers = 7;
elseif strcomp(model_name, 'resnet18')
    layers = {'relu', 'maxpool', 'layer1.0', 'layer1.1', ...
              'layer2.0', 'layer2.1', 'layer3.0', 'layer3.1', ...
              'layer4.0', 'layer4.1', 'avgpool'};
    num_layers = 11;
elseif strcomp(model_name, 'resnet50')
    layers = {'relu', 'maxpool', 'layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', ...
              'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0', 'layer3.1', ...
              'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0', ...
              'layer4.1', 'layer4.2', 'avgpool'};
     num_layers = 19;
elseif strcomp(model_name, 'resnet101')
    layers = {'relu', 'maxpool', 'layer1.0', 'layer1.1', 'layer1.2', ...
              'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0', ...
              'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', ...
              'layer3.6', 'layer3.7', 'layer3.8', 'layer3.9', 'layer3.10', ...
              'layer3.11', 'layer3.12', 'layer3.13', 'layer3.14', 'layer3.15', ...
              'layer3.16', 'layer3.17', 'layer3.18', 'layer3.19', 'layer3.20', ...
              'layer3.21', 'layer3.22', 'layer4.0', 'layer4.1', 'layer4.2', 'avgpool'};
     num_layers = 36;
elseif strcomp(model_name, 'cornet-s')
    layers = {'V1', 'V2', 'V4', 'IT', 'decoder.avgpool'};
    num_layers = 5;
elseif strcmp(model_name, 'slowfast_full1')
    layers = {'blocks.0.multipathway_blocks.0', 'blocks.0.multipathway_blocks.1',...
             'blocks.1.multipathway_blocks.0.res_blocks.0','blocks.1.multipathway_blocks.0.res_blocks.1',...
             'blocks.1.multipathway_blocks.0.res_blocks.2','blocks.1.multipathway_blocks.1.res_blocks.0',...
             'blocks.1.multipathway_blocks.1.res_blocks.1','blocks.1.multipathway_blocks.1.res_blocks.2',...
             'blocks.2.multipathway_blocks.0.res_blocks.0','blocks.2.multipathway_blocks.0.res_blocks.1',...
             'blocks.2.multipathway_blocks.0.res_blocks.2','blocks.2.multipathway_blocks.0.res_blocks.3',...
             'blocks.2.multipathway_blocks.1.res_blocks.0','blocks.2.multipathway_blocks.1.res_blocks.1',...
             'blocks.2.multipathway_blocks.1.res_blocks.2','blocks.2.multipathway_blocks.1.res_blocks.3',...
             'blocks.3.multipathway_blocks.0.res_blocks.0','blocks.3.multipathway_blocks.0.res_blocks.1',...
             'blocks.3.multipathway_blocks.0.res_blocks.2','blocks.3.multipathway_blocks.0.res_blocks.3',...
             'blocks.3.multipathway_blocks.0.res_blocks.4','blocks.3.multipathway_blocks.0.res_blocks.5',...
             'blocks.3.multipathway_blocks.1.res_blocks.0','blocks.3.multipathway_blocks.1.res_blocks.1',...
             'blocks.3.multipathway_blocks.1.res_blocks.2','blocks.3.multipathway_blocks.1.res_blocks.3',...
             'blocks.3.multipathway_blocks.1.res_blocks.4','blocks.3.multipathway_blocks.1.res_blocks.5',...
             'blocks.4.multipathway_blocks.0.res_blocks.0','blocks.4.multipathway_blocks.0.res_blocks.1',...
             'blocks.4.multipathway_blocks.0.res_blocks.2','blocks.4.multipathway_blocks.1.res_blocks.0',...
             'blocks.4.multipathway_blocks.1.res_blocks.1','blocks.4.multipathway_blocks.1.res_blocks.2',...
             'blocks.5','blocks.6.proj'};
     num_layers = 36;
end
rois = {'Early', 'MidVent', 'MidLat', 'MidPar', 'Ventral', 'Lateral', 'Parietal'};

%% load noise ceiling maps
data_dir = sprintf('%s/ppdata/subj%02d/nativesurface/betas_fithrf_GLMdenoise_RR/',nsd_datalocation('betas'),subjix);  
full_nc = struct([]);
for h = 1:length(hemis)
    nc = load_mgh([data_dir  sprintf('%s.nc_3trials.mgh',hemis{h})]);
    full_nc(1).(hemis{h}) = nc./100;
    %nc = load(sprintf('/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/local_data/freesurfer/%s/%s_split_half.mat', subjid,hemis{h}));
    %full_nc(1).(hemis{h}) = nc.mean';
end

%% load subject-to-subject fits
main_ROIs = {'Early', 'Midventral', 'Midlateral', 'Midparietal', 'Ventral', 'Lateral', 'Parietal'}; 
source_roi = 'streams_shrink20';
target_roi = 'streams_shrink10';
n_source_voxels = 5000;
num_splits = 1;
s2s = struct([]);
for h = 1:length(hemis)
    for l = 1:length(main_ROIs)
        s2s(h).(main_ROIs{l})= h5read(fullfile(s2s_path,[subjid, '_', hemis{h},'_', source_roi,  '_to_', target_roi, '_', num2str(num_splits), 'splits__subsample_', num2str(n_source_voxels), 'voxels_subsamptype1.hdf5']), ['/',main_ROIs{l}]);
    end
end


for h = 1:length(hemis)
    roivals = cvnloadmgz(sprintf('/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/data/nsddata/freesurfer/%s/label/%s.%s.mgz',subjid, hemis{h}, roi_name));  % load in an existing file?
    streams_trim = roivals(roivals ~= 0);
    
    num_vox = length(s2s(h).('Ventral'));

    for v = 1:num_vox
        
        if streams_trim(v) == 1
            s2s_matching_fit = s2s(h).('Early')(v);
        elseif streams_trim(v) == 2
            s2s_matching_fit = s2s(h).('Midventral')(v);
        elseif streams_trim(v) == 3
            s2s_matching_fit = s2s(h).('Midlateral')(v);
        elseif streams_trim(v) == 4
            s2s_matching_fit = s2s(h).('Midparietal')(v);
        elseif streams_trim(v) == 5
            s2s_matching_fit = s2s(h).('Ventral')(v);
        elseif streams_trim(v) == 6
            s2s_matching_fit = s2s(h).('Lateral')(v);
        elseif streams_trim(v) == 7
            s2s_matching_fit = s2s(h).('Parietal')(v);
        end    
        
        metrics(h).s2s_by_matching_roi(v) = s2s_matching_fit;
    end   
end

%% load model fits
fits = struct([]);
for h = 1:length(hemis)
    for l = 1:length(layers)
        key = strrep(layers{l}, '.', '_');
        if strcmp(model_name, 'slowfast_full1')
            fits(h).(key)= h5read(fullfile(fits_path,['by_layer_', subjid, '_', hemis{h},'_', roi_name,  '_', model_name, '_', method, '_subsample_', num2str(subsample), '_', num2str(CV), 'CV_1pretraining_fits.hdf5']), ['/',layers{l}]);
        else
            fits(h).(key)= h5read(fullfile(fits_path,[subjid, '_', hemis{h},'_', roi_name,  '_', model_name, '_', method, '_subsample_', num2str(subsample), '_', num2str(CV), 'CV_fits.hdf5']), ['/',layers{l}]);
        end
    end
end


%% calculate best layer for each voxel

for h = 1:length(hemis)

    key = strrep(layers{1}, '.', '_');
    num_vox = length(fits(h).(key));

    for v = 1:num_vox
        rs_by_layer = [];
        for l = 1:length(layers)
            key = strrep(layers{l}, '.', '_');
            rs_by_layer = [rs_by_layer fits(h).(key)(v)];
        end
        
        metrics(h).best_layer_by_vox(v) = find(rs_by_layer == max(rs_by_layer));
        metrics(h).max_rs(v) = max(rs_by_layer);   
    end   
end

%% format to work with full flat map using streams rois
idx = struct([]);
best_layer = []; best_rs = []; corrected_rs = []; total_nc = []; match_s2s = []; corrected_by_s2s = [];
join = [];
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
    
    temp = zeros(length(full_nc(1).(hemis{h})),1)';
    temp(idx(1).(hemis{h})) = metrics(h).max_rs./full_nc.(hemis{h})(idx(1).(hemis{h}))';
    corrected_rs = [corrected_rs temp];
    
    temp = zeros(length(full_nc(1).(hemis{h})),1)';
    temp(idx(1).(hemis{h})) = full_nc.(hemis{h})(idx(1).(hemis{h}))';
    total_nc = [total_nc temp];
    
    temp = zeros(length(full_nc(1).(hemis{h})),1)';
    temp(idx(1).(hemis{h})) = metrics(h).s2s_by_matching_roi;
    match_s2s = [match_s2s temp];
        
    temp = zeros(length(full_nc(1).(hemis{h})),1)';
    temp(idx(1).(hemis{h})) = metrics(h).max_rs./metrics(h).s2s_by_matching_roi;
    corrected_by_s2s = [corrected_by_s2s temp];
end

%% order results by ROI
byroi = struct([]);
for r = 1:length(rois)
    
    byroi(r).best_layer = best_layer(join == r);
    byroi(r).best_rs = best_rs(join == r);
    byroi(r).corrected_rs = corrected_rs(join == r);
end

%% plotting
extraopts = {'roiname',{'streams'},'roicolor',{'k'},'drawroinames',false, 'roiwidth', 2};
[rawimg,Lookup,rgbimg] = cvnlookup(subjid,10,best_layer',[.001,num_layers], jet(num_layers), .001,[],1,extraopts);
colormap(jet(num_layers))
colorbar('TickLabels', layers)
imwrite(rgbimg,sprintf('%s/%s_%s_%s_subsamp%s_best_layer.png',results_path,model_name,roi_name, subjid, num2str(subsample)));
%%
extraopts = {'roiname',{'streams'},'roicolor',{'k'},'drawroinames',false, 'roiwidth', 2};
[rawimg,Lookup,rgbimg] = cvnlookup(subjid,10,best_rs',[0,.7], hot(256), .001,[],1,extraopts);
imwrite(rgbimg,sprintf('%s/%s_%s_%s_subsamp%s_max_rs.png',results_path,model_name,roi_name, subjid, num2str(subsample)));
%%
extraopts = {'roiname',{'streams'},'roicolor',{'k'},'drawroinames',false, 'roiwidth', 2};
[rawimg,Lookup,rgbimg] = cvnlookup(subjid,10,corrected_rs',[0,1], hot(256), 0.001,[],1,extraopts);
imwrite(rgbimg,sprintf('%s/%s_%s_%s_subsamp%s_corrected_rs.png',results_path,model_name,roi_name,subjid, num2str(subsample)));
%%
extraopts = {'roiname',{'streams'},'roicolor',{'k'},'drawroinames',false, 'roiwidth', 2};
[rawimg,Lookup,rgbimg] = cvnlookup(subjid,10,total_nc',[0,1], hot(256), 0.001,[],1,extraopts);
imwrite(rgbimg,sprintf('%s/%s_%s_%s_just_nc.png',results_path,model_name,roi_name,subjid));
% 
%%
extraopts = {'roiname',{'streams'},'roicolor',{'k'},'drawroinames',false, 'roiwidth', 2};
[rawimg,Lookup,rgbimg] = cvnlookup(subjid,10,match_s2s',[0,1], hot(256), 0.001,[],1,extraopts);
imwrite(rgbimg,sprintf('%s/%s_%s_%s_just_s2s.png',results_path,model_name,roi_name,subjid));
%%
extraopts = {'roiname',{'streams'},'roicolor',{'k'},'drawroinames',false, 'roiwidth', 2};
[rawimg,Lookup,rgbimg] = cvnlookup(subjid,10,corrected_by_s2s',[0,1], hot(256), 0.001,[],1,extraopts);
imwrite(rgbimg,sprintf('%s/%s_%s_%s_corrected_by_s2s.png',results_path,model_name,roi_name,subjid));

close all