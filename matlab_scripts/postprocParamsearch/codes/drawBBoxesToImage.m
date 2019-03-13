function im = drawBBoxesToImage(inputImage, bboxes)
% bboxes 

% [h,w,d] = size(inputImage);

% bboxes(:,[1 3]) = bboxes(:,[1 3])/1024*h;
% bboxes(:,[2 4]) = bboxes(:,[2 4])/1024*w;

f = figure(1); 
a = axes('Parent', f);
imagesc(inputImage, 'Parent', a); hold on;
for i=1:size(bboxes,1)
    rectangle('Position', [bboxes(i,1), bboxes(i,2), bboxes(i,3)-bboxes(i,1), bboxes(i,4)-bboxes(i,2)],...
              'EdgeColor', [1 0 0], 'LineWidth', 2, 'Parent', a);
end

fr = getframe(a);
im = frame2im(fr);