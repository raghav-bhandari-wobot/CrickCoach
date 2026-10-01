@echo off
REM Runs the analyser on the sample videos on the GPU and logs everything to out\debug_log.txt
call .venv\Scripts\activate
if not exist out mkdir out
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" > out\debug_log.txt 2>&1
if not exist "videos\Sample 29_0-180.mp4" python tools\trim.py "videos\Sample 29.mp4" --start 0 --dur 180 >> out\debug_log.txt 2>&1
python analyse.py "videos\Sample 7.mp4" --no-show >> out\debug_log.txt 2>&1
python analyse.py "videos\Sample 29_0-180.mp4" --no-show >> out\debug_log.txt 2>&1
echo Done. >> out\debug_log.txt
type out\debug_log.txt | more
