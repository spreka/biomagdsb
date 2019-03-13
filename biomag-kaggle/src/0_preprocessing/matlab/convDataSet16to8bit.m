% Converts the uint16 tiff images into an one channel 8 bit png with the same name
root = '/home/biomag/etasnadi/input/histo/'; 
result_dir = strcat(root, 'out/');
cont = dir(strcat(root, '*.tif')); 
nfiles = size(cont); 
nfiles = nfiles(1); 
disp(nfiles);
for i = 1:nfiles
    [~, fname, ext] = fileparts(cont(i).name);
    in_fname = strcat(root, fname, ext);
    mkdir(result_dir);
    out_fname = strcat(result_dir, fname, '.png');
    
    disp(in_fname);
    disp(out_fname);
    %imwrite(uint8(imadjust(imread(in_fname))/256), out_fname);
    imwrite(imread(in_fname), out_fname);
end