@echo off
rem Eenmalig uitvoeren (dubbelklikken) op CrushMini.
rem 1) "Backtest Server": start de backtest server (poort 9000) bij het aanmelden.
rem 2) "Server Watchdog": kijkt elke 2 minuten of de servers op 5000, 5055, 5056, 8000 en 9000 antwoorden
rem    en start een gestopte server opnieuw met zijn eigen startcommando.

if not exist C:\watchdog mkdir C:\watchdog
copy /Y "%~dp0watchdog.ps1" C:\watchdog\watchdog.ps1 >nul

rem De taak "Backtest Server" bestaat al (aangemaakt per gebruiker, zonder adminrechten): dan niet opnieuw maken.
schtasks /query /tn "Backtest Server" >nul 2>&1 || schtasks /create /tn "Backtest Server" /sc onlogon /tr "C:\backtest-server\start-backtest-server.cmd" /rl limited /f
schtasks /create /tn "Server Watchdog" /sc minute /mo 2 /tr "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\watchdog\watchdog.ps1" /rl limited /f

echo.
echo Klaar. Controle:
schtasks /query /tn "Backtest Server"
schtasks /query /tn "Server Watchdog"
echo.
echo Logboek van de watchdog: C:\watchdog\watchdog.log
pause
