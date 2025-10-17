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
echo Starting training with Hydra defaults...
python train.py

REM Example override usage (uncomment to customize)
REM python train.py training.batch_size=16 data.num_frames=32

REM Check if training completed successfully
if errorlevel 1 (
    echo Training failed! Check error messages above.
) else (
    echo Training completed successfully!
)

echo.
pause