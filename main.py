from flask import Flask, request, abort
import os
import json
import psycopg2
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

app = Flask(__name__)

#環境変数取得
YOUR_CHANNEL_ACCESS_TOKEN = 'Z0YxYGEsYeu1Pzt40kvWY3rLDnxIg/JMneyX3IheisMKm1PVLLYHFhVAwn7svL73AzOF99SZgtmhoZ2QRvr9ou9tgTwZ3MQf0Cj5lBYjDhJPQholfik0z53eCYjrdcMiJfPh316PM3Onzdh0yFOxawdB04t89/1O/w1cDnyilFU='
YOUR_CHANNEL_SECRET = 'ac9df2376fe7721c2dde07a52d54c1af'

line_bot_api = LineBotApi(YOUR_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(YOUR_CHANNEL_SECRET)

@app.route("/")
def hello_world():
    return "hello world!"
def get_connection():
    dsn="host=ec2-35-172-73-125.compute-1.amazonaws.com port=5432 dbname=d1imih4v4olpv3 user=hgxrgiljdbmgur password=4bc62cfb1c5a966928e6263b84d180a10ba951c913ade0f3966288b6e602a4d5"
    return psycopg2.connect(dsn)
def get_response_message(mes_from):
    with get_connection() as conn:
        print('database connected successfullt')
        with conn.cursor() as cur:
            try:
                sql="INSERT INTO test(content) VALUES(%s);"
                cur.execute(sql,(mes_from,))
                return "succeed"
            except:
                return "exception"
    return "???"

            
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        text=get_response_message(event.message.text))

if __name__ == "__main__":
#    app.run()
    port = int(os.getenv("PORT",5432))
    app.run(host="0.0.0.0", port=port)