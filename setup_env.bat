@echo off
REM ============================================================================
REM AutoYOLO Trainer - cree le venv .venv et installe toutes les dependances.
REM
REM   setup_env.bat          PyTorch avec CUDA 12.6 (defaut, comme requirements.txt)
REM   setup_env.bat --cpu    PyTorch CPU uniquement
REM
REM Equivalent de : py -3.12 scripts\setup_env.py [--cpu]
REM ============================================================================
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "INDEX=cu126"
if /i "%~1"=="--cpu" set "INDEX=cpu"

echo === AutoYOLO Trainer : installation de l'environnement ===
echo Dossier : %CD%
echo Index PyTorch : %INDEX%
echo.

REM --- 1. Trouver Python 3.12 (lanceur py), sinon Python par defaut ----------
set "PY=py -3.12"
%PY% -c "import sys" >nul 2>&1
if not errorlevel 1 goto :python_ok

echo [!] Python 3.12 introuvable (lanceur "py -3.12").
echo     Installez-le : winget install Python.Python.3.12
set "PY=py -3"
py -3 -c "import sys" >nul 2>&1
if errorlevel 1 goto :no_python
echo [!] Utilisation de Python par defaut : le projet cible Python 3.12.
echo.

:python_ok
%PY% --version
echo.

REM --- 2. Creer le venv ------------------------------------------------------
if exist ".venv\Scripts\python.exe" goto :venv_ok
echo [..] Creation du venv .venv
%PY% -m venv .venv
if errorlevel 1 goto :error

:venv_ok
set "VPY=.venv\Scripts\python.exe"

REM --- 3. Outils de packaging -----------------------------------------------
echo.
echo [..] Mise a jour de pip, setuptools, wheel
"%VPY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error

REM --- 4. PyTorch d'abord, depuis le bon index (evite la version CPU par defaut)
echo.
echo [..] Installation de PyTorch (%INDEX%)
"%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/%INDEX%
if errorlevel 1 goto :error

REM --- 5. Le reste des dependances ------------------------------------------
echo.
echo [..] Installation de requirements.txt
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

REM --- 6. Verification -------------------------------------------------------
echo.
"%VPY%" -c "import torch, ultralytics; print('torch', torch.__version__, '| ultralytics', ultralytics.__version__); print('CUDA disponible :', torch.cuda.is_available())"
if errorlevel 1 (
    echo [X] PyTorch ne se charge pas. Verifiez l'espace disque libre et relancez le script.
    goto :error
)

echo.
echo === Termine. Pour activer l'environnement : .venv\Scripts\activate.bat ===
pause
exit /b 0

:no_python
echo [X] Aucun interpreteur Python detecte. Installez Python 3.12 puis relancez.
pause
exit /b 1

:error
echo.
echo [X] Echec de l'installation (voir les messages ci-dessus).
pause
exit /b 1
