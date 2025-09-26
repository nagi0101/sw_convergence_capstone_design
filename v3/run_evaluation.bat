@echo off
echo Starting VideoMAE Baseline Evaluation...
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

REM Check if checkpoint exists
if not exist "results\checkpoints\best_model.pth" (
    echo Error: Checkpoint not found at results\checkpoints\best_model.pth
    echo Please train the model first using run_training.bat
    pause
    exit /b 1
)

REM Run evaluation
echo Starting evaluation...
python evaluate.py --config config.yaml --checkpoint results\checkpoints\best_model.pth --output results\evaluation.yaml

REM Check if evaluation completed successfully
if errorlevel 1 (
    echo Evaluation failed! Check error messages above.
) else (
    echo Evaluation completed successfully!
    echo Results saved to results\evaluation.yaml
)

echo.
pause