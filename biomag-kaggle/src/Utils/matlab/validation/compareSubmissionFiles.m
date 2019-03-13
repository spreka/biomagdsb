%compare files

fMatlab = fopen('submission.csv','r');
fPython = fopen('submission_py.csv','r');

lineM = fgetl(fMatlab);
lineP = fgetl(fPython);

c = 1;
while ischar(lineM) && ischar(lineP)
    if ~strcmp(lineM(1:end-1),lineP)
        disp('Matlab line:');
        disp(lineM);
        disp('Python line:');
        disp(lineP);        
        c = c+1;
    end
    if c>5
        break;
    end
    lineM = fgetl(fMatlab);
    lineP = fgetl(fPython);
end

fclose(fMatlab);
fclose(fPython);