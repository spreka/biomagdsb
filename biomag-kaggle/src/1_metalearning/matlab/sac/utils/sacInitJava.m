function sacInitJava(SACROOTin)

if nargin == 1

    javaclasspathfunction(SACROOTin)

else
    
    global SACROOT
    SACROOTtemp = SACROOT; %#ok<*NODEF>
    global CommonHandles;
    CHTemp = CommonHandles;
    global GlobalHandles;
    GHTemp = GlobalHandles;

    javaclasspathfunction(SACROOTtemp)

    global SACROOT %#ok<*REDEF> % necessary to recover from a javaclasspath
                                % because all globals are cleared (including CommonHandles)   
    SACROOT = SACROOTtemp; %#ok<NASGU>
    global CommonHandles; 
    CommonHandles = CHTemp;
    global GlobalHandles;
    GlobalHandles = GHTemp;
end








function javaclasspathfunction(ROOT)

global logFID %#ok<*TLEV>
logFIDtemp = logFID;

javaclasspathstr = [ROOT '/thirdParty/Weka/weka.jar'];
if isempty(ismember(javaclasspath('-dynamic'), javaclasspathstr))
    javaclasspath([ROOT '/thirdParty/Weka/weka.jar']);
end

global logFID 
logFID = logFIDtemp; %#ok<NASGU>