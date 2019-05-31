%% validation code here
function [valid_p] = validation(model, xvalid_feat)
valid_feat = xvalid_feat;
[~, pred] = feed_foward(valid_feat, model);
[~, pred] = max(pred);
valid_p = pred';
end