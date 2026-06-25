@echo off
:: Locate xelatex.exe
set XELATEX=xelatex
where xelatex >nul 2>nul
if %errorlevel% neq 0 (
    if exist "%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\xelatex.exe" (
        set XELATEX="%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"
    )
)

echo Compiling main.tex...
%XELATEX% -interaction=nonstopmode main.tex
%XELATEX% -interaction=nonstopmode main.tex
echo Done! Output saved as main.pdf.
pause
