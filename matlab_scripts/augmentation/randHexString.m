function rString = randHexString(sLength)

s = 'abcdef0123456789';

%find number of random characters to choose from
numRands = length(s); 
%generate random string
rString = s( ceil(rand(1,sLength)*numRands) );