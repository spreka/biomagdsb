function correctedImg = correctWithUnet(inSegm, probMap, erRad, dilRad)
%tunes segmentation boundary with the help of UNet probabilities. erRad and
%dilRad is the maximal allowed distance of changing object contours inwards and
%outwards respectively.

inSegm = double(inSegm);
probMap = double(probMap > 32768);

maskSmall = double(erodeLabelledMasks(inSegm, erRad));
maskBig = double(dilateLabelledMasks(inSegm, dilRad));

ring = double(maskBig-maskSmall) .* probMap;

correctedImgBW = (ring+maskSmall)>0;

[~, labels] = bwdist( inSegm>0 );

correctedImg = inSegm(labels);
correctedImg(~correctedImgBW) = 0;

% figure(1);
% subplot(1,4,1); imagesc(inSegm); title('in');
% subplot(1,4,2); imagesc(double(inSegm>0)-probMap); title('inSegmbinarized - probmap binarized');
% subplot(1,4,3); imagesc(correctedImg); title('corrected');
% subplot(1,4,4); imagesc(correctedImg-inSegm); title('diff');
%
%% old code from Peter
%
% out = ring+maskSmall;
% outFinal = out * 0;
% 
% index = 1;
% for j=1:max(out(:))
%     
%     blank = out * 0;
%     pos = find(out == j);
%     blank(pos) = 1;
%     labelledBlank = bwlabel(blank, 4);
%     stats = regionprops(blank, 'Area');
%     if ~isempty(stats)
%         [maxv, maxi] = max(stats.Area);
%         if maxv > 7
%             outPos = find(labelledBlank == maxi);
%             outFinal(outPos) = index;
%             index = index + 1;
%         end
%     end
% end 

% figure(2);
% subplot(1,4,1); imagesc(inSegm); title('in');
% subplot(1,4,2); imagesc(double(inSegm>0)-probMap); title('inSegmbinarized - probmap binarized');
% subplot(1,4,3); imagesc(outFinal); title('corrected');
% subplot(1,4,4); imagesc(outFinal-inSegm); title('diff');
% 
% max(reshape(abs(outFinal-correctedImg),[],numel(outFinal)))

