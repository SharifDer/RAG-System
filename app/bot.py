# bot.py
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import requests

RAG_ENDPOINT = "http://localhost:8000/query"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Ask me anything!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    response = requests.post(RAG_ENDPOINT, json={"query": user_input})
    await update.message.reply_text(response.json()["answer"])

app = ApplicationBuilder().token("8114472992:AAHAVrrgqH5RC2joIKz5-0Gnlzg0_BXz6HQ").build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
app.run_polling()
