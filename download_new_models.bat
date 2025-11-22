@echo off
echo Downloading NEW quantized models...

cd backend\data\models\quantized

echo.
echo [1/2] Downloading Qwen 2.5 3B (1.9GB)...
curl -L --progress-bar -o qwen2.5-3b-instruct-q4.gguf ^
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf

echo.
echo [2/2] Downloading Phi-3 Mini 3.8B (2.3GB)...
curl -L --progress-bar -o phi-3-mini-4k-instruct.Q4_K_M.gguf ^
  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf

echo.
echo ✅ All models downloaded!
echo.
dir
pause