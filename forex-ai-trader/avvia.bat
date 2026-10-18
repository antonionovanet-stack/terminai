@echo off
rem Doppio clic per avviare (modalita simulata). Opzioni: avvia.bat -Ordini  /  -Evolvi  /  -Analisi
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0avvia.ps1" %*
pause
