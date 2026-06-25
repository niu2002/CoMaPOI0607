@echo off
:: Locate xelatex.exe
set XELATEX=xelatex
where xelatex >nul 2>nul
if %errorlevel% neq 0 (
    if exist "%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\xelatex.exe" (
        set XELATEX="%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"
    )
)

echo ===================================================
echo 1. Compiling ccf_a_hero_poi_paper.tex (Pass 1)...
echo ===================================================
%XELATEX% -interaction=nonstopmode ccf_a_hero_poi_paper.tex

echo ===================================================
echo 2. Compiling ccf_a_hero_poi_paper.tex (Pass 2)...
echo ===================================================
%XELATEX% -interaction=nonstopmode ccf_a_hero_poi_paper.tex

echo ===================================================
echo 3. Compiling ccf_a_hero_poi_paper_overleaf_fixed.tex (Pass 1)...
echo ===================================================
%XELATEX% -interaction=nonstopmode ccf_a_hero_poi_paper_overleaf_fixed.tex

echo ===================================================
echo 4. Compiling ccf_a_hero_poi_paper_overleaf_fixed.tex (Pass 2)...
echo ===================================================
%XELATEX% -interaction=nonstopmode ccf_a_hero_poi_paper_overleaf_fixed.tex

echo ===================================================
echo 5. Compiling ccf_a_hero_poi_paper_en.tex in english directory (Pass 1)...
echo ===================================================
pushd ..\english
%XELATEX% -interaction=nonstopmode ccf_a_hero_poi_paper_en.tex

echo ===================================================
echo 6. Compiling ccf_a_hero_poi_paper_en.tex in english directory (Pass 2)...
echo ===================================================
%XELATEX% -interaction=nonstopmode ccf_a_hero_poi_paper_en.tex
popd

echo ===================================================
echo Compilation complete! PDFs generated.
echo ===================================================

