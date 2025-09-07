@echo off
echo Creating portable fraud detection system...
echo.

REM Download portable Python
if not exist "python" (
    echo Downloading Portable Python...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/winpython/winpython/releases/download/4.2.20231111/Winpython64-3.9.13.0dot.exe' -OutFile 'python_installer.exe'"
    python_installer.exe /VERYSILENT /DIR=python
    del python_installer.exe
)

REM Install packages
echo Installing required packages...
python\python.exe -m pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib

REM Create run script
echo Creating run script...
(
echo @echo off
echo echo Starting Fraud Detection App...
echo cd /d "%%~dp0"
echo python\python.exe -m streamlit run app_enhanced_nosmote.py
echo pause
) > run_app.bat

echo.
echo Setup complete! Double-click 'run_app.bat' to start the app.
pause