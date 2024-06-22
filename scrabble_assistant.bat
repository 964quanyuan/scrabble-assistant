@echo off
echo hello mutherfucker!
start /B python -c "from sounds.play_audio import greet; greet()"
start /B python observe.py
start "" "C:\Program Files\e2eSoft\iVCAM\iVCAM.exe"
start /min explorer "C:\Users\31415\Videos\iVCam"
timeout /t 1 /nobreak >nul
echo loading program...
echo please wait patiently. it may take up to 15 seconds (or 15 minutes if ur on yo mama's laptop)
timeout /t 1 /nobreak >nul
python -c "from sounds.play_audio import rick_entry; rick_entry()"
