@ECHO off
SETLOCAL ENABLEEXTENSIONS
SetLocal EnableDelayedExpansion
:: batch script to generate custom validation list for training

:: parameters are coming
set root_dir=%~dp0

if "%1"=="" (
    echo "Usage: %0 validation_folder"
    exit /B 1
)

set images_dir=%1

if "%2"=="" (
    echo "using default validation names folder: %root_dir%\matlab_scripts\"
    set "target_dir=%root_dir%\matlab_scripts\"
) else (
    set target_dir=%2
)

echo "GENERATING VALIDATION NAMES:"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%target_dir%\generateValidation\');generateValidationCustom('%target_dir%','%images_dir%');exit;"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during cell size estimation"
    exit /B 2
)
echo "GENERATING VALIDATION NAMES DONE"