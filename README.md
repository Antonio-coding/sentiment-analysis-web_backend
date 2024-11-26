### Sentiment Analysis Web: README

# 📊 **Sentiment Analysis Web**

O projeto **Sentiment Analysis Web** é uma aplicação para análise de sentimentos baseada em inteligência artificial. Ele avalia comentários inseridos pelo usuário e prevê a quantidade de estrelas que o comentário representa em uma escala de 1 a 5. Este projeto foi desenvolvido como parte da disciplina de **Inteligência Artificial** no curso de **Sistemas para Internet** da **Universidade Católica de Pernambuco (UNICAP)**.

---

## 🚀 **Funcionalidades**

- Análise preditiva de sentimentos com base em comentários em português ou inglês.
- Classificação de comentários em uma escala de 1 a 5 estrelas.
- Frontend responsivo desenvolvido com **Next.js** e **Tailwind CSS**.
- Backend robusto em **Python**, utilizando bibliotecas modernas como **FastAPI** e **Transformers**.
- Integração entre frontend e backend via API.

---

## 📦 **Estrutura do Projeto**

- **Frontend**: Desenvolvido com Next.js. Hospedado na **Vercel**.
- **Backend**: Implementado em Python utilizando FastAPI, rodando em ambiente local com suporte ao túnel gerado pelo **ngrok**.

Repositórios:

- [Frontend no GitHub](https://github.com/Antonio-coding/sentiment-analysis-web_frontend)
- [Backend no GitHub](https://github.com/Antonio-coding/sentiment-analysis-web_backend)

---

## 🛠 **Como Executar o Projeto**

### **1. Modo Local**

#### **Frontend**

1. Clone o repositório do frontend:
   ```bash
   git clone https://github.com/Antonio-coding/sentiment-analysis-web_frontend
   cd sentiment-analysis-web_frontend
   ```
2. Instale as dependências:
   ```bash
   npm install
   ```
3. Inicie o servidor de desenvolvimento:
   ```bash
   npm run dev
   ```
4. Acesse a aplicação no navegador em: `http://localhost:3000`.

#### **Backend**

1. Clone o repositório do backend:
   ```bash
   git clone https://github.com/Antonio-coding/sentiment-analysis-web_backend
   cd sentiment-analysis-web_backend
   ```
2. Instale o **Python** (versão 3.8 ou superior) no seu computador.
3. Instale as dependências do backend:
   ```bash
   pip install fastapi uvicorn transformers pyngrok torch nest-asyncio
   ```
4. Configure o **ngrok** para gerar o túnel:
   - Crie uma conta no [ngrok](https://ngrok.com/).
   - Configure seu authtoken:
     ```bash
     ngrok authtoken <seu_authtoken>
     ```
   - Inicie o túnel na porta padrão:
     ```bash
     ngrok http 8000
     ```
5. Atualize o caminho do túnel no frontend:
   - Substitua a variável de API no arquivo de ambiente do frontend (`.env.local`) com o URL gerado pelo ngrok, por exemplo:
     ```env
     NEXT_PUBLIC_API_URL=https://<caminho-do-tunel>.ngrok.io
     ```
6. Inicie o servidor do backend:
   ```bash
   uvicorn main:app --reload
   ```

---

### **2. Modo Online com Deploy na Vercel**

- Para rodar o frontend online, hospede-o no **Vercel**.
- Certifique-se de que o backend está rodando em uma máquina local e o caminho gerado pelo **ngrok** foi configurado na variável de ambiente da Vercel.
- Sempre atualize o caminho do túnel toda vez que reiniciar o ngrok.

---

## 👥 **Equipe**

| Nome                                | GitHub                                              | Avatar                                                           |
| ----------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------- |
| Antônio Thiago Neto                 | [Antonio-coding](https://github.com/Antonio-coding) | ![Avatar](https://avatars.githubusercontent.com/u/112974950?v=4) |
| Danilo Fernandes Rodrigues da Silva | [danilozxz](https://github.com/danilozxz)           | ![Avatar](https://avatars.githubusercontent.com/u/149892834?v=4) |
| Carlos Eduardo Bezerra de Lima      | [Eduard0177](https://github.com/Eduard0177)         | ![Avatar](https://avatars.githubusercontent.com/u/131684798?v=4) |
| Danilo Santana                      | [DaniloSantos31](https://github.com/DaniloSantos31) | ![Avatar](https://avatars.githubusercontent.com/u/130468402?v=4) |
| Gabriel Pereira Araújo              | [GabrielP02](https://github.com/GabrielP02)         | ![Avatar](https://avatars.githubusercontent.com/u/115106954?v=4) |
| Alysson Rafael Freire Silva Santos  | [Alysson04](https://github.com/Alysson04)           | ![Avatar](https://avatars.githubusercontent.com/u/131741122?v=4) |

---

## 📝 **Licença**

Este projeto foi desenvolvido para fins acadêmicos e está sob licença MIT. Sinta-se à vontade para explorar, modificar e usar o código, respeitando os direitos dos autores.
