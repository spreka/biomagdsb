function score = optimizerFunc(params, gtMap, smallScaleImagesMap, bigScaleImagesMap, sumProbMap, fid)

%FMINCON
%{
scaleThrsh = round(params(1) * 100);
probThresh = round(params(2) * 100000);
erosionRadius = round(params(3) * 10);
dilationRadius = round(params(4) * 10);
minSize = round(params(5) * 100);
%}

%FMINSEARCH
%{
scaleThrsh = round(params(1));
probThresh = round(params(2));
erosionRadius = round(params(3));
dilationRadius = round(params(4));
minSize = round(params(5));
minOverlap = params(6) / 1000;
%}

%GA
scaleThrsh = params(1) / 100;
probThresh = params(2);
erosionRadius = params(3);
dilationRadius = params(4);
minSize = params(5);
minOverlap = params(6) / 10000;
maxV = params(7);
cArea = params(8);
medianSize = params(9);

areaThresh = 10000;

outFinalImageMap = mergeUnetAndAll(smallScaleImagesMap, bigScaleImagesMap, sumProbMap, scaleThrsh, probThresh, erosionRadius, dilationRadius, minSize, minOverlap, maxV, cArea, areaThresh, medianSize);
%writeSegmentation(outFinalImageMap, '/media/baran/LinuxData/Downloads/Challange/Optimizer/temp_outFinal', '.tiff');

scoreMap = evaluation2(gtMap, outFinalImageMap);

%FMINSEARCH
%score = 1 - mean(cell2mat(scoreMap.values));
%fprintf('Score for corrected segmentations (scaleTh:%d, probTh:%d, er:%dpx, dil:%dpx, min:%dpx): %0.3f\n', scaleThrsh, probThresh, erosionRadius, dilationRadius, minSize, 1.0 - score);
%fprintf(fid,'Score for corrected segmentations (scaleTh:%d, probTh:%d, er:%dpx, dil:%dpx, min:%dpx): %0.3f\n', scaleThrsh, probThresh, erosionRadius, dilationRadius, minSize, 1.0 - score);

%FMINCON
%score = mean(cell2mat(scoreMap.values));
%fprintf('Score for corrected segmentations (scaleTh:%d, probTh:%d, er:%dpx, dil:%dpx, min:%dpx): %0.3f\n', scaleThrsh, probThresh, erosionRadius, dilationRadius, minSize, score);
%fprintf(fid2,'Score for corrected segmentations (scaleTh:%d, probTh:%d, er:%dpx, dil:%dpx, min:%dpx): %0.3f\n', scaleThrsh, probThresh, erosionRadius, dilationRadius, minSize, score);

%GA
meanVal = mean(cell2mat(scoreMap.values));
score = 1.0 - meanVal;
fprintf('Score for corrected segmentations (scaleTh:%0.2f, probTh:%d, er:%dpx, dil:%dpx, minSize:%dpx, minOverlap:%0.4f, maxV:%d, cArea:%d, median:%d): %0.4f\n', scaleThrsh, probThresh, erosionRadius, dilationRadius, minSize, minOverlap, maxV, cArea, medianSize, meanVal);
fprintf(fid,'Score for corrected segmentations (scaleTh:%0.2f, probTh:%d, er:%dpx, dil:%dpx, minSize:%dpx, minOverlap:%0.4f, maxV:%d, cArea:%d, median:%d): %0.4f\n', scaleThrsh, probThresh, erosionRadius, dilationRadius, minSize, minOverlap, maxV, cArea, medianSize, meanVal);

end
