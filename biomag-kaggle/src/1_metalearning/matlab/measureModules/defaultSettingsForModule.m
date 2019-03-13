function [ settings ] = defaultSettingsForModule( moduleName )
%defaultSettingsForModule
%   gives back the cellarray of default settings

r = 1; c = 1;
switch moduleName
    case 'MeasureObjectIntensity'       
        settings{r,c} = '-'; c=c+1;
        settings{r,c} = 'FakeObject'; c=c+1;
        settings{r,c} = 'Do not use'; c=c+1;
        settings{r,c} = 'Do not use'; c=c+1;
        settings{r,c} = 'Do not use'; c=c+1;
        settings{r,c} = 'Do not use'; c=c+1;
        settings{r,c} = 'Do not use';
    case 'MeasureTexture'
        settings{r,c} = '-'; c=c+1;
        settings{r,c} = 'FakeObject'; c=c+1;
        settings{r,c} = 'Do not use'; c=c+1;
        settings{r,c} = 'Do not use'; c=c+1;
        settings{r,c} = 'Do not use'; c=c+1;
        settings{r,c} = 'Do not use'; c=c+1;
        settings{r,c} = 'Do not use'; c=c+1;
        settings{r,c} = '3'; 
end


end

