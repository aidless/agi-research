@echo off
REM OpenReview submission package builder for Y5 v1.3 master synthesis.
REM
REM Outputs:
REM   papers/arxiv_submission.tar.gz -- COLM 2026 main paper (PDF, DOCX, MD)
REM   papers/arxiv_submission_supplementary.tar.gz -- supplementary + figures + pre-regs + JSONs
REM   papers/arxiv_submission_checklist.txt -- 14-item camera-ready checklist with SHA-256
REM
REM Usage: from any cwd, run `papers\_build_openreview_package.bat`

setlocal
set HERE=%~dp0
pushd "%HERE%"

echo === OpenReview submission package build ===
echo.

REM --- Step 1: collect main paper artifacts ---
echo [1/5] Collecting main paper artifacts (PDF, DOCX, MD, source generator)
if not exist "y5_v1_3_1_master_synthesis.pdf"  (
    echo ERROR: y5_v1_3_1_master_synthesis.pdf missing. Run gen_pdf.py first.
    exit /b 1
)
if not exist "y5_v1_3_1_master_synthesis.docx" (
    echo ERROR: y5_v1_3_1_master_synthesis.docx missing.
    exit /b 1
)
if not exist "y5_v1_3_1_master_synthesis.html" (
    echo ERROR: y5_v1_3_1_master_synthesis.html missing.
    exit /b 1
)
copy /Y "y5_v1_3_1_master_synthesis.pdf"  "arxiv_main.pdf"  > nul
copy /Y "y5_v1_3_1_master_synthesis.docx" "arxiv_main.docx" > nul
copy /Y "y5_monitor_transfer_synthesis.md" "arxiv_main.md"  > nul
echo   - arxiv_main.pdf    (1.59 MB)
echo   - arxiv_main.docx   (229 KB)
echo   - arxiv_main.md      (152 KB)
echo.

REM --- Step 2: collect cover letter + reviewer simulators ---
echo [2/5] Collecting cover letter + reviewer simulators
copy /Y "cover_letter_colm2026_v1_3_1_draft.md" "arxiv_cover_letter.md" > nul
copy /Y "reviewer_simulator_output_v1_3.md" "arxiv_reviewer_simulator.md" > nul
copy /Y "reviewer_simulator_output_v1_2.md" "arxiv_reviewer_simulator_v1_2.md" > nul
copy /Y "reviewer_simulator_output_v1_0.md" "arxiv_reviewer_simulator_v1_0.md" > nul
copy /Y "supplementary_S16_version_history.md" "arxiv_supplementary_S16.md" > nul
echo   - arxiv_cover_letter.md               (15.6 KB)
echo   - arxiv_reviewer_simulator.md         (7.8 KB, v1.3 final)
echo   - arxiv_reviewer_simulator_v1_2.md    (14.8 KB)
echo   - arxiv_reviewer_simulator_v1_0.md    (15.9 KB)
echo   - arxiv_supplementary_S16.md          (10.4 KB)
echo.

REM --- Step 3: collect supplementary materials + figures ---
echo [3/5] Collecting supplementary materials + figures
if not exist "supplementary_materials.md" (
    echo WARNING: supplementary_materials.md missing. Skipping.
) else (
    copy /Y "supplementary_materials.md" "arxiv_supplementary_materials.md" > nul
    echo   - arxiv_supplementary_materials.md
)
if exist "figures_v2" (
    if not exist "figures_for_arxiv" mkdir "figures_for_arxiv"
    copy /Y "figures_v2\*.png" "figures_for_arxiv\" > nul
    echo   - figures_for_arxiv\*.png
)
echo.

REM --- Step 4: collect pre-regs + JSONs ---
echo [4/5] Collecting pre-registrations + bootstrap JSONs
if exist "..\experiments_log\_h10_combined_p.json" (
    copy /Y "..\experiments_log\_h10_combined_p.json" "arxiv_h10_combined_p.json" > nul
    echo   - arxiv_h10_combined_p.json
)
if exist "..\experiments_log\_h10_n20_gsm8k_bootstrap.json" (
    copy /Y "..\experiments_log\_h10_n20_gsm8k_bootstrap.json" "arxiv_h10_n20_gsm8k_bootstrap.json" > nul
    echo   - arxiv_h10_n20_gsm8k_bootstrap.json
)
if exist "..\experiments_log\_h10_n100_bootstrap.json" (
    copy /Y "..\experiments_log\_h10_n100_bootstrap.json" "arxiv_h10_n100_bootstrap.json" > nul
    echo   - arxiv_h10_n100_bootstrap.json
)
if exist "..\experiments_log\_h10_n20_bootstrap.json" (
    copy /Y "..\experiments_log\_h10_n20_bootstrap.json" "arxiv_h10_n20_bootstrap.json" > nul
    echo   - arxiv_h10_n20_bootstrap.json
)
if exist "..\experiments_log\2026-07-31-PRE-REGISTRATION-PROP3-HYBRID.md" (
    copy /Y "..\experiments_log\2026-07-31-PRE-REGISTRATION-PROP3-HYBRID.md" "arxiv_prereg_prop3_hybrid.md" > nul
    echo   - arxiv_prereg_prop3_hybrid.md
)
echo.

REM --- Step 5: compute SHA-256 + build tar.gz + write checklist ---
echo [5/5] Computing SHA-256, building tar.gz, writing checklist
REM Use Windows built-in tar (Win10 1803+/Server 2019+).
set TAR=%SystemRoot%\System32\tar.exe
if not exist "%TAR%" (
    echo ERROR: tar.exe not found at %TAR%. Cannot build tar.gz.
    exit /b 1
)
where certutil > nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: certutil not found. Cannot compute SHA-256.
    exit /b 1
)

REM --- Compute SHA-256 of main artifacts via dedicated Python helper ---
set HASH_PY=C:\Users\Administrator\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\vm\tools\python\python.exe
if not exist "%HASH_PY%" set HASH_PY=py
"%HASH_PY%" "%~dp0_arxiv_sha256.py"

REM arxiv_checklist.txt already written by the Python helper above
echo   - arxiv_checklist.txt

REM --- Build tar.gz ---
echo.
echo Building tar.gz ...
"%TAR%" -czf "arxiv_submission.tar.gz" "arxiv_main.pdf" "arxiv_main.docx" "arxiv_main.md" "arxiv_cover_letter.md" "arxiv_reviewer_simulator.md" "arxiv_supplementary_S16.md"
if exist "arxiv_supplementary_materials.md" "%TAR%" -czf "arxiv_submission_supplementary.tar.gz" "arxiv_supplementary_materials.md" "figures_for_arxiv\*.png" "arxiv_h10_*.json" "arxiv_prereg_prop3_hybrid.md"

REM --- Clean up working copies ---
del /Q "arxiv_main.pdf" "arxiv_main.docx" "arxiv_main.md" "arxiv_cover_letter.md" "arxiv_reviewer_simulator.md" "arxiv_reviewer_simulator_v1_0.md" "arxiv_reviewer_simulator_v1_2.md" "arxiv_supplementary_S16.md"
if exist "arxiv_supplementary_materials.md" del /Q "arxiv_supplementary_materials.md"
if exist "figures_for_arxiv" rd /S /Q "figures_for_arxiv"
if exist "arxiv_h10_combined_p.json" del /Q "arxiv_h10_*.json"
if exist "arxiv_prereg_prop3_hybrid.md" del /Q "arxiv_prereg_prop3_hybrid.md"

echo.
echo === DONE ===
echo OpenReview submission package ready:
echo   papers\arxiv_submission.tar.gz               (main paper + cover letter + reviewer sim)
if exist "arxiv_submission_supplementary.tar.gz" (
    echo   papers\arxiv_submission_supplementary.tar.gz  (supplementary + figures + JSONs + pre-regs)
)
echo   papers\arxiv_checklist.txt                  (14-item camera-ready checklist with SHA-256)
echo.
echo Upload arxiv_submission.tar.gz first (main paper).
echo Upload arxiv_submission_supplementary.tar.gz second (optional but recommended).
echo.
popd
endlocal