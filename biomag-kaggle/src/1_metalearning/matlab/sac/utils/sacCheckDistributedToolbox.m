function pll = sacCheckDistributedToolbox()

global logFID

pll = license('test', 'distrib_computing_toolbox');

if pll
    sacLog(3,logFID,' Distributed Computing Toolbox check: OK\n');
else
    sacLog(2,logFID,' Warning: Distributed Computing Toolbox license NOT FOUND.\n It is STRONGLY recommended that SAC is run using the Distributed Computing Toolbox.\n');
end