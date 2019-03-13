function borderRemoved = clearBorderObjects(labelledImage, varargin)
% clearBorderObjects removes objects at the image border.
% The selected borders to clear can be set by a 1x4 binary vector for the 4
% borders respectively in the following order: [right top left bottom].

if nargin<2
    clearDirections = [1 1 1 1];
else
    clearDirections = varargin{1};
end

map = 0:max(labelledImage(:));
if clearDirections(2)
    map(labelledImage(1,:)+1)=0;
end
if clearDirections(4)
    map(labelledImage(end,:)+1)=0;
end
if clearDirections(3)
    map(labelledImage(:,1)+1)=0;
end
if clearDirections(1)
    map(labelledImage(:,end)+1)=0;
end

borderRemoved = map(labelledImage+1);