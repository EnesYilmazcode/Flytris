@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Flytris overnight run
if not exist runs\overnight mkdir runs\overnight
set LOG=runs\overnight\overnight.log

rem Turn off QuickEdit for this window so a stray click cannot pause the run.
powershell -NoProfile -Command "$s='[DllImport(\"kernel32.dll\")]public static extern IntPtr GetStdHandle(int h);[DllImport(\"kernel32.dll\")]public static extern bool GetConsoleMode(IntPtr h,out uint m);[DllImport(\"kernel32.dll\")]public static extern bool SetConsoleMode(IntPtr h,uint m);';$k=Add-Type -MemberDefinition $s -Name K -Namespace Q -PassThru;$h=$k::GetStdHandle(-10);$m=0;[void]$k::GetConsoleMode($h,[ref]$m);[void]$k::SetConsoleMode($h,($m -band (-bnot 0x40)) -bor 0x80)" >nul 2>&1

echo Flytris is training. Leave this window open and the laptop plugged in.
echo Progress: runs\train_after\train.log   Stage log: %LOG%
echo Please do not click inside this window.
echo [%date% %time%] overnight run started>> %LOG%

call :stage train "python -X utf8 scripts\train.py --head afterstate --pop 32 --elites 5 --validate-every 10 --until 03:00"
if exist scripts\evaluate.py call :stage evaluate "python -X utf8 scripts\evaluate.py"
call :stage tournament "python -X utf8 scripts\tournament.py"

echo [%date% %time%] overnight run finished>> %LOG%
echo Done. See %LOG%
pause
exit /b 0

:stage
for /L %%i in (1,1,3) do (
    echo [!date! !time!] %~1 attempt %%i>> %LOG%
    %~2 >> runs\overnight\%~1.log 2>&1
    if !errorlevel! equ 0 (
        echo [!date! !time!] %~1 ok>> %LOG%
        exit /b 0
    )
    echo [!date! !time!] %~1 exited with !errorlevel!>> %LOG%
)
echo [!date! !time!] %~1 FAILED after 3 attempts>> %LOG%
exit /b 1
