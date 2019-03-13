function outImg = augment_colorFlip(outImg, invert, colorFlip)

% invert image
if invert
    outImg = 255 - outImg;
end

% flip colors randomly and invert images randomly with 50% likelyness
if colorFlip
    order = randperm(3);
    if rand()>0.5
        oITmp(:,:,1) = 255 - outImg(:,:,order(1));
    else
        oITmp(:,:,1) = outImg(:,:,order(1));
    end
    if rand()>0.5
        oITmp(:,:,2) = 255 - outImg(:,:,order(2));
    else
        oITmp(:,:,2) = outImg(:,:,order(2));
    end
    if rand()>0.5
        oITmp(:,:,3) = 255 - outImg(:,:,order(3));
    else
        oITmp(:,:,3) = outImg(:,:,order(3));
    end
    outImg  = oITmp;
    
end
