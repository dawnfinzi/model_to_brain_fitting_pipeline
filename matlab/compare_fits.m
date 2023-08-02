%% Load in model fitting results and plot on cortical surface

close all 
clear all

local_data_path = '/share/kalanit/biac2/kgs/projects/Dawn/NSD/local_data';
fits_path = '/share/kalanit/biac2/kgs/projects/Dawn/NSD/results/fits';
oak_fits_path = '/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/results/fits';
results_path = '/share/kalanit/biac2/kgs/projects/Dawn/NSD/results';

hemis = {'rh'}; %'lh', 
subjid = 'subj02';  
t_thresh = .1;
roi_name = 'streams_shrink10';

layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7'};
rois = {'Early', 'MidVent', 'MidLat', 'MidPar', 'Ventral', 'Lateral', 'Parietal'};


%% load model fits
fits = struct([]);
for h = 1:length(hemis)
    for l = 1:length(layers)
        fits(h).(layers{l})= h5read(fullfile(fits_path,[subjid, '_', hemis{h},'_', roi_name, '_alexnet_25c_fits.hdf5']), ['/',layers{l}]);
    end
end

%% load comparison model fits
comp_fits = struct([]);
for h = 1:length(hemis)
    for l = 1:length(layers)
        comp_fits(h).(layers{l})= h5read(fullfile(fits_path,[subjid, '_', hemis{h},'_', roi_name, '_alexnet_25c_fullfeats_fits.hdf5']), ['/',layers{l}]);
    end
end

%% calculate best layer for each voxel

for h = 1:length(hemis)
    
    num_vox = length(fits(h).(layers{1}));

    for v = 1:num_vox
        rs_by_layer = [];
        for l = 1:length(layers)
            rs_by_layer = [rs_by_layer fits(h).(layers{l})(v)];
        end
        
        metrics(h).best_layer_by_vox(v) = find(rs_by_layer == max(rs_by_layer));
        metrics(h).max_rs(v) = max(rs_by_layer);   
        
        %% comparison fits
        comp_rs_by_layer = [];
        for l = 1:length(layers)
            comp_rs_by_layer = [comp_rs_by_layer comp_fits(h).(layers{l})(v)];
        end
        
        comp_metrics(h).best_layer_by_vox(v) = find(comp_rs_by_layer == max(comp_rs_by_layer));
        comp_metrics(h).max_rs(v) = max(comp_rs_by_layer);   
    end   
end

%% plot nicely
h=1; %one hemi for now

fontSize = 11; titleSize = 14;
f(1) = niceFig([.1 .1 .55 .9],fontSize,2);

figure(f(1));
    
s = scatter(metrics(h).max_rs, comp_metrics(h).max_rs ,.5,'b'); hold on;
%s.MarkerEdgeAlpha = 0.25;
hline = refline([1 0]);
hline.Color = 'k';
hline.LineStyle = '-';

xl = xlim; xlim([0 .65]); yl = ylim; ylim([0 .65]);
set(gca,'TickDir','out');
xlabel('Subsample Fit (R^2)','FontSize',fontSize); ylabel(['All Features Fit (R^2)'],'FontSize',fontSize); 
%xlabel('Other subject fits (R^2)','FontSize',fontSize); ylabel(['Model fits (R^2)'],'FontSize',fontSize); 
axis square; 
title('Fit comparison across all voxels');

figure(f(1));
niceSave(results_path,['/subj02_subsampleVSallfeats'],[],[],{'png' 'svg'});
