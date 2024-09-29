from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters.command import Command
from aiogram import types
from aiogram.enums import ParseMode
import sys
import asyncio
import logging
import aiohttp

TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"  # Замените на ваш токен бота

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher(parse_mode=ParseMode.MARKDOWN)

# URL API (localhost если бот на одном сервере с api)
API_URL = "http://0.0.0.0:8000/predict"


@dp.message(Command("start"))
async def start(message: types.Message):
    # user_id = message.from_user.id
    start_message = (
        "👋 Привет! Я - интеллектуальный помощник оператора поддержки RUTUBE\n\n"
    )

    await message.answer(start_message)


@dp.message(F.text)
async def handle_message(message: types.Message):
    question = message.text
    data = {"question": question}

    # запрос к FastAPI
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, json=data) as response:
            if response.status != 200:
                await message.reply("Произошла ошибка. Попробуйте позже.")
                return

            # ответ от API
            result = await response.json()
            answer = result.get("answer", "Не удалось получить ответ.")
            class_1 = result.get("class_1", "Неизвестно")
            class_2 = result.get("class_2", "Неизвестно")

            # ответное сообщение
            response_text = (
                f"Ответ: {answer}\n"
                f"Класс 1: {class_1}\n"
                f"Класс 2: {class_2}"
            )
            await message.reply(response_text, parse_mode=ParseMode.HTML)


async def main():
    await dp.start_polling(bot, parse_mode=ParseMode.HTML)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
