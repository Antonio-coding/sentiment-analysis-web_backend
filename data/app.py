import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pyngrok import ngrok
import nest_asyncio
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")

# Configurar o ngrok com o token de autenticação
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Configuração do modelo
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Configuração do FastAPI
app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de entrada
class Review(BaseModel):
    review: str

# Função para predição
def predict_sentiment(review: str) -> dict:
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
    probabilities_dict = {f"{i+1} estrela(s)": round(prob * 100, 2) for i, prob in enumerate(probabilities)}
    return probabilities_dict

# Rota para predição
@app.post("/predict")
def predict(review: Review):
    probabilities = predict_sentiment(review.review)
    return {"probabilities": probabilities}

# Aplicação do nest_asyncio para Colab ou execução contínua
nest_asyncio.apply()

if __name__ == "__main__":
    # Configuração do túnel ngrok
    public_url = ngrok.connect(8000)
    print(f"API pública acessível em: {public_url}")

    # Execução do servidor Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
