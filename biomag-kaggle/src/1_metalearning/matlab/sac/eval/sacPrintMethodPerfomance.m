function sacPrintMethodPerfomance(methodString,logFID,criteria,values,P)
%
% P is the performance index!!!
%

    sacLog(3, logFID, '    P:%1.4f   ', P);
    for m = 1:numel(criteria)
        sacLog(3,logFID,'%s:%1.4f ', criteria{m}, values(m));
    end
    sacLog(3, logFID, '[%s]\n', methodString);
    
% else
%     
%     
%     
%     
% end