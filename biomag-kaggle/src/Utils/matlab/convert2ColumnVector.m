function [ columnVector ] = convert2ColumnVector( vector )
%convert2ColumnVector It is often a problem that a vector is not in right
%dimensions, and the programmer should remember whether the vector was a
%column or a row vector. (even knowing that in MatLab the vector is column by
%default, but that's not true for all the cases)
%   This function converts a vector to a Column vector whatever it was
%   before (column or row). The pair of this function is the
%   convert2RowVector converts to row from arbitrary vector.

if (size(vector,1)<size(vector,2))
   columnVector = vector'; 
else
   columnVector = vector;
end

end

