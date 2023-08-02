clear all
close all

roi_name = 'streams';
subjix=1
gradtype = 'y';

[raw, Lookup, rgbimg] = cvnlookup(sprintf('subj%02d',subjix),13);
l = size(Lookup{1,1}.imglookup);
r = size(Lookup{1,2}.imglookup);
if strcmp(gradtype, 'x')
    grad_shape = l(2);
    left_shape = l(1);
    right_shape = r(1);
elseif strcmp(gradtype, 'y')
    grad_shape = l(1);
    left_shape = l(2);
    right_shape = r(2);
end

%ROIs
left = cvnloadmgz(sprintf('/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/local_data/freesurfer/subj0%s/lh.%s.mgz',num2str(subjix), roi_name));  % load in an existing file?
right = cvnloadmgz(sprintf('/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/local_data/freesurfer/subj0%s/rh.%s.mgz',num2str(subjix), roi_name));  % load in an existing file?

gradient = linspace(0,1,grad_shape);

full = repmat(gradient', 1, left_shape);
temp = zeros(size(left)); 
lookup = Lookup{1,1}.imglookup;
for row = 1:left_shape
    if strcmp(gradtype, 'x')
        look = lookup(row,:);
        grad = full(:,row);
    elseif strcmp(gradtype, 'y')
        look = lookup(:,row);
        grad = full(:,row);
    end
    temp(look) = grad;
end
left_grad = temp;

if strcmp(gradtype, 'x')
    grad_shape = r(2);
    gradient = linspace(1,0,grad_shape);
end
full = repmat(gradient', 1, right_shape);
temp = zeros(size(right)); 
lookup = Lookup{1,2}.imglookup;
for row = 1:right_shape
    if strcmp(gradtype, 'x')
        look = lookup(row,:);
        grad = full(:,row);
    elseif strcmp(gradtype, 'y')
        look = lookup(:,row);
        grad = full(:,row);
    end
    temp(look) = grad;
end
right_grad = temp;

extraopts = {'roiname',{'streams'},'roicolor',{'k'},'drawroinames',false, 'roiwidth', 2};

[raw, Lookup, rgbimg] = cvnlookup(sprintf('subj%02d',subjix),13,[left_grad; right_grad], [], [], [], [], 1, extraopts); 

save_path = sprintf('~/oak_dtn/Dawn/NSD/results/spacetorch/gradient_keys/subj%02d_%sgradient.mat', subjix,gradtype);
save(save_path, 'left_grad', 'right_grad')
       
%close all