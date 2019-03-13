function fts=extractFeaturesEasy(src,imName)
% Extracts easy intensity-features from image to determine image class as
% in either one of the following:
% - fluorescent
% - tissue
% - brightfield
% The function reads the image as [src '\' imName] or if the image is given
% as input, just parses it.
% 
    
    if nargin==1 && ~ischar(src)
        % image is already read
        img=im2double(src);
    else
        img=im2double(imread(fullfile(src,imName)));
    end
    
    isColorIm=numel(size(img))>2; % color image
    if isColorIm
        imgg=rgb2gray(img);
        gray=imgg(:);
        color=cell(1,3);
        qc=zeros(3,5);
        stc=zeros(1,3);
        for c=1:3
            imgc=img(:,:,c);
            color{1,c}=imgc(:);
            qc(c,:)=quantile(color{1,c},[0,.25,.5,.75,1]);
            stc(1,c)=std(color{1,c});
        end
    else
        gray=img(:);
    end
    q(1,:)=quantile(gray,[0,.25,.5,.75,1]);
    st=std(gray);
    
    if ~isColorIm
        for c=1:3
            qc(c,:)=q(:);
            stc(1,c)=st;
        end
    end
    
    fts(1,:)=[qc(1,:) qc(2,:) qc(3,:) q stc st];
end