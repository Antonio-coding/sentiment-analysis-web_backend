from transformers import pipeline

pipe = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pyngrok import ngrok
import nest_asyncio
import uvicorn

# Carregar o modelo pré-treinado e o tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Definir o FastAPI app
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

# Adicione o middleware CORS ao seu app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens. Ajuste para maior segurança, se necessário.
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos HTTP (GET, POST, etc.).
    allow_headers=["*"],  # Permite todos os cabeçalhos.
)

# Modelo de entrada
class Review(BaseModel):
    review: str

# Função para realizar a predição
def predict_sentiment(review: str) -> dict:
    # Tokenizar a entrada
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Obter a predição do modelo
    outputs = model(**inputs)

    # Converter logits para probabilidades usando softmax
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()

    # Criar um dicionário com as porcentagens por classe
    probabilities_dict = {f"{i+1} estrela(s)": round(prob * 100, 2) for i, prob in enumerate(probabilities)}

    return probabilities_dict

# Rota para a predição
@app.post("/predict")
def predict(review: Review):
    probabilities = predict_sentiment(review.review)
    return {"probabilities": probabilities}

# Aplicar o nest_asyncio para rodar o servidor no Colab
nest_asyncio.apply()

# Iniciar o servidor Uvicorn em uma thread separada
if __name__ == "__main__":
    # Abrir o túnel ngrok na porta 8000
    public_url = ngrok.connect(8000)
    print(f"API pública acessível em: {public_url}")

    # Rodar o servidor
    uvicorn.run(app, host="0.0.0.0", port=8000)
