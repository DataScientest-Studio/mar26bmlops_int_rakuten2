#!/bin/bash

# Chrome über Windows starten + neue Session + Tabs
powershell.exe -Command \
"Start-Process 'chrome.exe' -ArgumentList '--new-window 
http://localhost:8080/home 
http://localhost:5000/ 
http://localhost:9001/ 
http://localhost:3000/ 
http://localhost:8501/ 
http://localhost:8000/docs/ 
http://localhost:9090/ 
http://localhost:5050/docs'"