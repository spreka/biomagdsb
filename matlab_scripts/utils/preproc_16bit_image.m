function outimg=preproc_16bit_image(in,out)
	if ischar(in)
		img=imread(in);
	else
		img=in;
	end
	imgg=double(img);
	tmp=((imgg-min(imgg(:)))*255)/(max(imgg(:))-min(imgg(:)));
	outimg=uint8(tmp);

	if nargin>1
		imwrite(outimg,out);
	end
end