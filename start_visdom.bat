@ECHO off
SETLOCAL ENABLEEXTENSIONS
SetLocal EnableDelayedExpansion

set pyVirtPath=%1
echo "python virtenv path: " %pyVirtPath%

call %pyVirtPath%\Scripts\activate
python -m visdom.server