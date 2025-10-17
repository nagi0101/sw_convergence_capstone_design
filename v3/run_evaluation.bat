@echo off
setlocal ENABLEDELAYEDEXPANSION
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

REM Parse optional --wandb flag and forward remaining Hydra overrides
set "ENABLE_WANDB=0"
set "CMD_ARGS= evaluation.checkpoint_path=results\checkpoints\best_model.pth"

:parse_args
if "%1"=="" goto args_done
if /I "%1"=="--wandb" (
    set "ENABLE_WANDB=1"
) else (
    set "CMD_ARGS=!CMD_ARGS! %1"
)
shift
goto parse_args

:args_done
if "%ENABLE_WANDB%"=="1" (
    set "CMD_ARGS=!CMD_ARGS! logging.wandb.enable=true"
)

REM Run evaluation
echo Starting evaluation...
python evaluate.py !CMD_ARGS!

REM Check if evaluation completed successfully
if errorlevel 1 (
    echo Evaluation failed! Check error messages above.
) else (
    echo Evaluation completed successfully!
    echo Results saved to results\evaluation.yaml
)

echo.
endlocal
pause