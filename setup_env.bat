@echo off
echo Setting up Python environment...

REM Remove existing virtual environments if they exist
if exist venv rmdir /s /q venv
if exist .venv rmdir /s /q .venv

REM Create a new virtual environment
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Install PyTorch and other dependencies
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-learn

echo Environment setup complete!