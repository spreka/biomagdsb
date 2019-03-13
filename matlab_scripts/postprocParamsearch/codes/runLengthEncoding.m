function nuclei = runLengthEncoding(maskImg)
    R = regionprops(maskImg,'PixelIdxList','BoundingBox','Area');
    N = length(R);
    
    nuclei = cell(N,1);
    i = 1;
    ii = 1;
    while i<=N
        if R(i).Area > 0
            j = 1;        
            currPix = R(i).PixelIdxList;
            runLengthCoding = zeros(1,2*(R(i).BoundingBox(3))); %this is just a good estimation
            c = 1;
            while j<=length(currPix)
                k = 1;          
                runLengthCoding(2*c-1) = currPix(j);            
                while (j<length(currPix) && currPix(j) == currPix(j+1)-1)
                    k = k+1;
                    j = j+1;
                end                  
                runLengthCoding(2*c) = k;
                c = c+1;
                j = j+1;
            end
            runLengthCoding(2*c-1:end) = []; % we need to delete extra columns
            nuclei{ii} = runLengthCoding;            
            ii = ii+1;
        end
        i = i+1;        
    end
    nuclei(ii:end) = [];
end