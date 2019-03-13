function out = augment_blur(in, scale)

% max 3-4

H = fspecial('gaussian',[round(scale * 5)+3 round(scale * 5)+3], scale);

out = imfilter(in,H,'replicate');
