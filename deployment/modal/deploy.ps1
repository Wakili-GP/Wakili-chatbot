Set-Location "d:\FOE\Senior 2\Graduation Project\Chatbot fastapi\Wakili-chatbot"

python -m pip install --upgrade pip
python -m pip install modal

modal token new

$GroqKey = (Get-Content .env | Select-String "^GROQ_API_KEY").ToString().Split("=")[1].Trim().Trim('"')
modal secret create wakili-secrets GROQ_API_KEY="$GroqKey" GROQ_MODEL_NAME="llama-3.3-70b-versatile"

modal deploy modal_app.py
