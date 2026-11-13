@echo off
cd /d "%~dp0"
"C:\Users\uSeR\env_isaaclab\Scripts\python.exe" scripts\play.py --num_envs 1 --viz kit --checkpoint ..\..\machine-skills\pick_place_policy\model_7998.pt %*
pause
