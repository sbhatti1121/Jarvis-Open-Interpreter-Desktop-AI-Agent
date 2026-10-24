# Jarvis AI Assistant for Windows

A PySide6 desktop UI that listens for **“Hey Jarvis”**, transcribes commands using Google's online speech recognition endpoint, sends the text to the installed Open Interpreter CLI, and speaks the response using Microsoft Edge TTS.

## Requirements

- Windows 10/11
- Python 3.10 or newer (64-bit recommended)
- Open Interpreter already installed and configured
- Working microphone and speakers
- Internet access for online speech recognition and TTS

## Run Commands

1. irm https://www.openinterpreter.com/install.ps1 | iex
2. pip install -r requirements.txt
3. python main.py


