clear all
close all

%% %%%%% DEFINE THE ROIS

subjix = 1;   % which subject
subjid = 'subj01';
cmap   = jet(256);   % colormap for ROIs

rng    = [0 7];      % should be [0 N] where N is the max ROI index
roilabels = {'early' 'midventral' 'midlateral' 'midparietal' 'ventral' 'lateral' 'parietal'};  % 1 x N cell vector of strings

lh_a1 = load_mgh([sprintf('%s/ppdata/subj%02d/nativesurface/betas_fithrf_GLMdenoise_RR/',nsd_datalocation('betas'),subjix) 'lh.nc_3trials.mgh']);  
rh_a1 = load_mgh([sprintf('%s/ppdata/subj%02d/nativesurface/betas_fithrf_GLMdenoise_RR/',nsd_datalocation('betas'),subjix) 'rh.nc_3trials.mgh']);  

mgznames = {'Kastner2015' {lh_a1 rh_a1} 'streams' 'flocfacestval' 'floc-faces' 'flocbodytval' 'floc-bodies' 'flocplacestval' 'floc-places' 'flocwordval'};  % quantities of interest (1 x Q)
crngs = {[0 25] [0 75] [0 7] [0 10] [0 10] [0 10] [0 10] [0 10] [0 10] [0 10]};  % ranges for the quantities (1 x Q)
cmaps = {jet(256) jet parula copper copper copper copper copper copper copper};  % colormaps for the quantities (1 x Q)
threshs = {0.5 [20] .5 2.7 .5 2.7 .5 2.7 .5 2.7};
%roivals = cvnloadmgz(sprintf('~/nsd/nsddata/freesurfer/%s/label/?h.floc-bodies.mgz',subjid));  % load in an existing file?
%roivals = zeros(length(lh_a1) + length(rh_a1),1);
%roivals(length(lh_a1)+1:end) = cvnloadmgz(sprintf('/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/data/nsddata/freesurfer/subj%02d/label/rh.streams.mgz', subjix)); %[];

% do it
cvndefinerois;