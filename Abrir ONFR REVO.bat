@echo off
setlocal
chcp 65001 >nul
title ONFR REVO
cd /d "%~dp0"

echo.
echo   ONFR REVO  -  Restore without altering.
echo.

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

if not exist ".venv\revo_instalado.txt" (
    "%VPY%" -m pip install --upgrade pip
    "%VPY%" -m pip install -r requirements.txt
    if errorlevel 1 goto error
    echo ok> ".venv\revo_instalado.txt"
)

rem --- 3. Modelos faciales (solo la primera vez) -------------------------
"%VPY%" setup_models.py
if errorlevel 1 goto error

rem --- 4. Acceso directo en el escritorio --------------------------------
if not exist "%USERPROFILE%\Desktop\ONFR REVO.lnk" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\ONFR REVO.lnk');" ^
      "$s.TargetPath='%~f0'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%SystemRoot%\System32\imageres.dll,72'; $s.Save()" >nul 2>nul
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
