@echo off
setlocal ENABLEDELAYEDEXPANSION
echo Starting VideoMAE Baseline Training (CPU Configuration)...
echo.
echo WARNING: This uses reduced settings for CPU testing.
echo For full training, use run_training.bat with a GPU.
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

REM Parse optional --wandb flag and forward remaining Hydra overrides
set "ENABLE_WANDB=0"
set "CMD_ARGS= --config-name config_cpu"

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

REM Run training with CPU config
echo Starting training with Hydra CPU configuration...
python train.py !CMD_ARGS!

REM Check if training completed successfully
if errorlevel 1 (
    echo Training failed! Check error messages above.
) else (
    echo Training completed successfully!
)

echo.
endlocal
pause