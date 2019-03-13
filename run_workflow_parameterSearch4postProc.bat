@ECHO off
SETLOCAL ENABLEEXTENSIONS
SetLocal EnableDelayedExpansion
:: batch script to run prediction with postprocessing

:: parameters are coming
set root_dir=%1

set matlab_scripts=%root_dir%\matlab_scripts

:: check inputs
echo "root_dir: " %root_dir%
echo "matlab_scripts: " %matlab_scripts%


::::::::::::::::::: Parameter searching :::::::::::::::::::
matlab -noFigureWindows -nodesktop -nosplash -minimize -wait -log -r "addpath(genpath(%matlab_scripts%)); rootDir=%root_dir%; startPostProcParamSearch(rootDir);exit"