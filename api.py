from flask import Flask, request
import json
from analise_sentimento import predict_sentiment

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/analise", methods=["POST"])
def analisar_sentimento():
    frase = json.loads(request.data)
    return predict_sentiment(frase["texto"])
    
app.run()