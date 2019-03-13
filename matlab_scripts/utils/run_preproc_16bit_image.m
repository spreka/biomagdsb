function run_preproc_16bit_image(src,dest,extOut)
% runs 16-bit preprocessing to generate 8-bit RGB images

exts={'png','tif','tiff','bmp','jpg','jpeg'};

if nargin<2
    fprintf('Usage: run_preproc_16bit_image(input_dir,output_dir)\n');
    return;
elseif nargin==2
    % default output file format
    extOut='.tiff';
elseif nargin==3
    % extOut comes from input
    if strcmp(extOut(1),'.')
        % ok
    else
        extOut=['.' extOut];
    end
    if ~any(contains(exts,extOut(2:end)))
        fprintf('Supported formats: png, tif, tiff, bmp, jpg, jpeg\n');
        return;
    end
end

if ~exist(dest,'dir')
    mkdir(dest);
end

l=[];
for e=1:numel(exts)
	l_e=dir(fullfile(src,['*.' exts{e}]));
	l=[l;l_e];
end
fprintf('found %d images in %s\n',numel(l),src);
for i=1:numel(l)
	[~,base,ext]=fileparts(l(i).name);
	in=fullfile(src,[base ext]);
	outName=fullfile(dest,[base extOut]);
	img=imread(in);
	if numel(size(img))==3
		% colour image
		for ch=1:3
			outimg(:,:,ch)=preproc_16bit_image(img(:,:,ch));
		end
	else
		% grayscale image
		outimg=preproc_16bit_image(img);
	end

	imwrite(outimg,outName);
end

end