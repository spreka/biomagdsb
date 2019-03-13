function relabelledImage = relabelImage(labelledImage, varargin)
%relabelImage assigns consecutive labels to objects in the input image. 
% If the second argument is true it works similar way to bwlabel method.

labelledImage = uint16(labelledImage);

if nargin<2
    preserveOrder = 0;
else
    preserveOrder = varargin{1};
end

if preserveOrder
    uvor1 = unique(labelledImage(:), 'stable');
else
    uvor1 = unique(labelledImage(:));
end
if uvor1(1) == 0
    uvor1(1) = [];
end

lut = uint16(zeros(1,2^16));
lut(uvor1+1) = 1:length(uvor1);

relabelledImage = intlut(labelledImage, lut);
