@echo off
echo Starting VideoMAE Baseline Training...
echo.

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo Failed to activate virtual environment!
    echo Please ensure .venv exists and contains Python installation.
    pause
    exit /b 1
)

echo Virtual environment activated.
echo.

REM Run training
echo Starting training with config.yaml...
python train.py --config config.yaml

REM Check if training completed successfully
if errorlevel 1 (
    echo Training failed! Check error messages above.
) else (
    echo Training completed successfully!
)

echo.
pause