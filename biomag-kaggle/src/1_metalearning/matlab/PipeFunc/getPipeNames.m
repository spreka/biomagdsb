function headers = getPipeNames( pipes )
%getPipeNames
%   Lists the currently available pipes

headers = cell(1,length(pipes));
for i=1:length(pipes)
    headers{i} = pipes{i}.name;
end

end

