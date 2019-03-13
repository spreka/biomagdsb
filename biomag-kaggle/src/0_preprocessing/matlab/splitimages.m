% splits the images in the given inpDir into w*h sized patches
% the result will be in the outDir/{image_fname_wo_ext}_x_y.{ext}
function splitimages(inpDir, outDir, w, h, ext)

    %w = 256;
    %h = 256;
    %ext = '.png';
    %root_dir = '/home/biomag/etasnadi/input/processed-dataset'

    conts = dir([inpDir, '/*', ext]);
    for fid=1:numel(conts)
        fname = conts(fid).name;

        [~, fnamewext, ext] = fileparts(fname);
        out_fname_pref = fullfile(outDir, fnamewext);
        im = imread(fullfile(inpDir, fname));
        size_im = size(im);
        xs = size_im(2);
        ys = size_im(1);

        xpatches = uint32(floor(xs/w));
        ypatches = uint32(floor(ys/h));
        disp(['mkdir' out_fname_pref ' xpatches: ' num2str(xpatches) ' ypatches: ' num2str(ypatches)]);
        mkdir(out_fname_pref);
        xshift = (xs-(xpatches*w))/2;
        yshift = (ys-(ypatches*h))/2;
        for xpatch=1:xpatches
            for ypatch = 1:ypatches
                xfrom = xshift + ((xpatch-1)*w+1);
                xto = xshift + (xpatch*w);
                yfrom = yshift + ((ypatch-1)*h+1);
                yto = yshift + (ypatch*h);
                disp(strcat(num2str(xfrom), '-', num2str(xto)));
                out_fname = fullfile(out_fname_pref, [fnamewext, '_', num2str(xpatch), '_', num2str(ypatch), ext]);
                disp(['imwrite ' out_fname]);
                imwrite(im(yfrom:yto, xfrom:xto,:), out_fname);
                %disp(strcat(num2str(xpatch), fullfile(out_dir, 'asd'), num2str(ypatch)));
            end
        end
    end
end