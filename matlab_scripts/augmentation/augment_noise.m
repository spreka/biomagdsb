function out = augment_noise(in, noiseChoice)

% noiseChose -> 1..3

grayIn = rgb2gray(in);

switch noiseChoice
    case 1
        sigmaInt = rand(1) * 0.02;
        grayNoisy = double(imnoise(grayIn, 'gaussian', 0.0, sigmaInt));
    case 2
        speckleInt = rand(1) * 4;  
        spInt = rand(1) * 0.05;
        grayNoisy = double(imnoise(imnoise(grayIn, 'speckle', speckleInt), 'salt & pepper', spInt));
    case 3
        speckleInt = rand(1) * 0.1;
        grayNoisy = double(imnoise(grayIn, 'speckle', speckleInt));
end


in = double(in);
onlyNoise = grayNoisy-double(grayIn); %only noise
gray3chNoise = repmat(onlyNoise, [1 1 3]);
out = uint8(gray3chNoise + in);
