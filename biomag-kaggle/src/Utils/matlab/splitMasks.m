function splitMasks(src,dest)
% Splits uint16 type mask images to individual binary mask files and
% creates masks folders under the image name's folder for them, based on
% original mask images found in src folder.
% src: folder of labelled masks (uint16)
% dest: folder of individual masks (doesn't need to exist yet)

    l=dir([src '*.png']);
    for imi=1:numel(l)
        dname=fullfile(dest,l(imi).name(1:end-4),'masks');
        if exist(dname,'dir')
            continue;
        else
            mkdir(dname);
            img=imread(fullfile(src,l(imi).name));
            maxi=max(img(:));
            for i=1:maxi
                b=img==i;
                imwrite(b,sprintf('%s%c%03d.png',dname,filesep,i));
            end
        end
    end
end