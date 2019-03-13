function maskOut = relabel(maskIn)

recode = unique(maskIn(:));

maskOut = maskIn * 0;

for i=1:numel(recode)
    pix = maskIn == recode(i);
    maskOut(pix) = i-1;
end

