# build_fixed.py
import PyInstaller.__main__
import os

print("Building standalone executable with fix...")

PyInstaller.__main__.run([
    'app_enhanced.py',
    '--onefile',
    '--add-data', 'data;data',
    '--add-data', 'models;models',
    '--name', 'FraudDetection',
    '--console',
    '--hidden-import', 'sklearn',
    '--hidden-import', 'pandas',
    '--hidden-import', 'numpy',
    '--hidden-import', 'streamlit',
    '--exclude-module', 'dis',  # This fixes the issue!
])

print("Build completed! Check the 'dist' folder.")