@echo off
rem Se ejecuta desde una copia temporal: asi este archivo se puede actualizar
rem (git pull) mientras se usa sin que Windows se lie al leerlo.
if not "%REVO_FROM_TEMP%"=="1" (
    set "REVO_FROM_TEMP=1"
    set "REVO_DIR=%~dp0"
    set "REVO_BAT=%~f0"
    copy /y "%~f0" "%TEMP%\revo_abrir.bat" >nul
    "%TEMP%\revo_abrir.bat"
    exit /b
)
setlocal
chcp 65001 >nul
title ONFR REVO
cd /d "%REVO_DIR%"

echo.
echo   ONFR REVO  -  Restore without altering.
echo.

rem --- 0. Ultima version (si hay internet) ----------------------------------
where git >nul 2>nul
if not errorlevel 1 if exist ".git" (
    echo Buscando la ultima version de REVO...
    set "GIT_TERMINAL_PROMPT=0"
    git pull --ff-only --quiet || echo   No se ha podido actualizar: se abre la version que ya tienes.
)

rem --- 1. Python ---------------------------------------------------------
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo No se ha encontrado Python. Instalalo desde https://www.python.org/downloads/
    echo marcando la casilla "Add python.exe to PATH" y vuelve a abrir este archivo.
    pause
    exit /b 1
)

rem --- 2. Entorno propio (solo la primera vez) ---------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Preparando REVO por primera vez. Puede tardar unos minutos...
    %PY% -m venv .venv
    if errorlevel 1 goto error
)
set "VPY=.venv\Scripts\python.exe"

rem librerias: se reinstalan solo si requirements.txt ha cambiado
if exist ".venv\revo_instalado_v2.txt" if not exist ".venv\requirements.instalado.txt" copy /y requirements.txt ".venv\requirements.instalado.txt" >nul
fc /b requirements.txt ".venv\requirements.instalado.txt" >nul 2>nul
if errorlevel 1 (
    echo Actualizando REVO. Puede tardar unos minutos...
    "%VPY%" -m pip install --upgrade pip
    "%VPY%" -m pip uninstall -y opencv-python-headless opencv-python >nul 2>nul
    "%VPY%" -m pip install -r requirements.txt
    if errorlevel 1 goto error
    copy /y requirements.txt ".venv\requirements.instalado.txt" >nul
)

rem --- 3. Modelos faciales (solo la primera vez) -------------------------
"%VPY%" setup_models.py
if errorlevel 1 goto error

rem --- 4. Acceso directo en el escritorio --------------------------------
if not exist "%USERPROFILE%\Desktop\ONFR REVO.lnk" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\ONFR REVO.lnk');" ^
      "$s.TargetPath='%REVO_BAT%'; $s.WorkingDirectory='%REVO_DIR%'; $s.IconLocation='%SystemRoot%\System32\imageres.dll,72'; $s.Save()" >nul 2>nul
)

rem --- 5. Abrir ----------------------------------------------------------
echo.
echo REVO se esta abriendo en el navegador. Deja esta ventana abierta mientras lo uses;
echo cierrala para apagar REVO.
echo.
set "REVO_OPEN_BROWSER=1"
"%VPY%" app.py
goto :eof

:error
echo.
echo Algo ha fallado. Copia el texto de esta ventana y pegaselo a Claude.
pause
exit /b 1
