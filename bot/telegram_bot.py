from __future__ import annotations
import subprocess
import asyncio
import logging
import os
import requests
from uuid import uuid4
from telegram import BotCommandScopeAllGroupChats, Update, constants , ReplyKeyboardMarkup, WebAppInfo
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, InlineQueryResultArticle
from telegram import InputTextMessageContent, BotCommand
from telegram.error import RetryAfter, TimedOut
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, \
    filters, InlineQueryHandler, CallbackQueryHandler, Application, ContextTypes, CallbackContext

from pydub import AudioSegment
import base64
from utils import is_group_chat, get_thread_id, message_text, wrap_with_indicator, split_into_chunks, \
    edit_message_with_retry, get_stream_cutoff_values, is_allowed, get_remaining_budget, is_admin, is_within_budget, \
    get_reply_to_message_id, add_chat_request_to_usage_tracker, error_handler, is_direct_result, handle_direct_result, \
    cleanup_intermediate_files
from openai_helper import OpenAIHelper, localized_text
from usage_tracker import UsageTracker
from functools import wraps
from telegram.ext import ConversationHandler
import telegram
from db import Database
# Определение состояния
from dotenv import load_dotenv

PHOTO, USER_INPUT, DRAW_IMAGE, EMAIL = range(4)
PROCESS_VOICE = 0
import httpx
import aiofiles
from pytonconnect import TonConnect
import re
import subprocess
import json
import os
import logging
from telegram import Update, BotCommand

import asyncio



env_path = "/root/geniusBro/chatgpt-telegram-bot/.env"

# Загрузка переменных
load_dotenv(dotenv_path=env_path)

# Теперь можно получить переменную окружения
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')



UPLOAD_PDF, ANALYZE_PDF = range(2)
logger = logging.getLogger(__name__)

from yookassa import Payment, Configuration

Configuration.account_id = '279257'
Configuration.secret_key = 'live_AnL1eJLWO3YWYED9ao7zr8MZB9ZVvbpZJlfiYkJSjww'



def create_and_save_payment_method(amount, return_url, email):
    payment = Payment.create({
        "amount": {
            "value": str(amount),
            "currency": "RUB"
        },
        "payment_method_data": {
            "type": "bank_card"
        },
        "confirmation": {
            "type": "redirect",
            "return_url": return_url
        },
        "capture": True,
        "description": "Оплата сервиса на месяц",
        "receipt": {
            "customer": {
                "email": email
            },
            "items": [
                {
                    "description": "Оплата подписки",
                    "quantity": "1",
                    "amount": {
                        "value": str(amount),
                        "currency": "RUB"
                    },
                    "vat_code": 1,  # НДС, соответствующий вашей стране/ситуации
                }
            ]
        },
        "save_payment_method": True  # Раскомментируйте, если требуется сохранять метод оплаты
    })

    return payment
    
    
async def check_payment_status(payment_id, context, user_id):
    db = Database('/root/geniusBro/user_wallets.db')  # Путь к вашей базе данных

    while True:
        payment = Payment.find_one(payment_id)
        if payment.status == "succeeded":
            # Оплата успешно проведена
            db.save_payment_method_id(user_id, payment.payment_method.id)
            await context.bot.send_message(chat_id=user_id, text="Платеж успешно проведен!")
            
            # Обновление статуса подписки в базе данных
            db.update_paid_subscription(user_id, True)

            break
        elif payment.status == "canceled":
            # Оплата отменена
            await context.bot.send_message(chat_id=user_id, text="Платеж был отменен.")
            break
        await asyncio.sleep(10)  # Проверка каждые 10 секунд





def create_auto_payment(user_id, amount):
    # Получаем email пользователя из базы данных
    user_email = db.get_user_email(user_id)

    # Получаем ID сохраненного способа оплаты
    payment_method_id = db.get_payment_method_id(user_id)

    if payment_method_id and user_email:
        payment = Payment.create({
            "amount": {
                "value": str(amount),
                "currency": "RUB"
            },
            "capture": True,
            "payment_method_id": payment_method_id,
            "description": "Автоплатеж за подписку",
            "receipt": {
                "customer": {
                    "email": user_email
                },
                # [Другие данные чека]
            }
        })
        return payment
    else:
        return None
        
        
        
async def execute_scheduled_payments():
    db = Database('/root/geniusBro/user_wallets.db')
    users = db.get_all_subscribed_users()  # Получаем всех пользователей с активной подпиской

    for user in users:
        user_id = user['user_id']
        payment = create_auto_payment(user_id, 350)  # Сумма подписки
        if payment and payment.status == 'succeeded':
            print(f"Платеж для пользователя {user_id} успешно проведен.")
        else:
            print(f"Не удалось провести платеж для пользователя {user_id}.")
        await asyncio.sleep(1)  # Пауза для предотвращения перегрузки сервера        


def has_paid_subscription(user_id):
    
    
    paid_subscription, _ = db.check_paid_subscription(user_id)
    return paid_subscription




async def is_subscribed(context: CallbackContext, user_id: int, channel: str) -> bool:
    try:
        chat_member = await context.bot.get_chat_member(chat_id=channel, user_id=user_id)
        return chat_member.status in ['member', 'administrator', 'creator']
    except Exception as e:
        logging.error(f"Error checking subscription: {e}")
        return False


def token_limit_check(func):
    @wraps(func)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id

        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

        tokens_month = self.usage[user_id].get_current_token_usage()[1]
        
        

        # Проверка, достиг ли пользователь бесплатного лимита в 30000 токенов
        if tokens_month >= 250000:
            if has_paid_subscription(user_id):
                # Пользователь оплатил подписку, проверяем лимит в 100000 токенов
                if tokens_month >= 100000:
                    await update.message.reply_text(
                        "Вы исчерпали месячный лимит токенов в 100000. Пожалуйста, произведите оплату для возобновления доступа к функциям бота."
                    )
                    return
            else:
                # Пользователь не оплатил подписку и достиг бесплатного лимита
                keyboard = [[InlineKeyboardButton("Оплатить", callback_data="pay_subscription")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(
                    "Вы достигли бесплатного лимита в 30000 токенов. Пожалуйста, произведите оплату для получения доступа к дополнительным функциям бота.",
                    reply_markup=reply_markup
                )
                return

        return await func(self, update, context, *args, **kwargs)
    
    return wrapper

def image_limit_check(func):
    @wraps(func)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id

        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

        images_month = self.usage[user_id].get_current_image_count()[1]
        logger.info(f"User ID: {user_id}, Images this month: {images_month}")

        if images_month >= 2:
            if not has_paid_subscription(user_id):
                keyboard = [[InlineKeyboardButton("Оплатить", callback_data="pay_subscription")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(
                    "Вы достигли лимита в 2 изображения. Пожалуйста, произведите оплату для получения доступа к дополнительным функциям бота.",
                    reply_markup=reply_markup
                )
                return

        return await func(self, update, context, *args, **kwargs)
    
    return wrapper

def subscription_required(handler):
    @wraps(handler)
    async def check_subscription(self, update: Update, context: CallbackContext, *args, **kwargs):
        user_id = update.effective_user.id
        if await is_subscribed(context, user_id, '@genius_bro_channel'):
            return await handler(self, update, context, *args, **kwargs)
        else:
            # Updated subscription message with hyperlink
            subscription_message = (
                "Для начала работы, пожалуйста, подпишитесь на [канал](https://t.me/genius_bro_channel/2). "
                "Это нужно, чтобы отсеять ботов, а также, чтобы мы оставались на связи в случае возможных блокировок."
            )
            await context.bot.send_message(chat_id=update.effective_chat.id, text=subscription_message, parse_mode='Markdown')
    return check_subscription


db = Database('/root/geniusBro/user_wallets.db')


     
class ChatGPTTelegramBot:

        
    def __init__(self, config: dict, openai: OpenAIHelper):
        """
        Initializes the bot with the given configuration and GPT bot object.
        :param config: A dictionary containing the bot configuration
        :param openai: OpenAIHelper object
        """
        self.config = config
        self.openai = openai
        bot_language = self.config['bot_language']
        self.web_app_url = "https://4290-91-107-125-14.ngrok-free.app"
        # Оставляем список команд пустым
        self.commands = []

    
        self.group_commands = [BotCommand(
            command='chat', description=localized_text('chat_description', bot_language)
        )] + self.commands
        self.disallowed_message = localized_text('disallowed', bot_language)
        self.budget_limit_message = localized_text('budget_limit', bot_language)
        self.usage = {}
        self.last_message = {}
        self.inline_queries_cache = {}
        self.connector = TonConnect(manifest_url='https://4290-91-107-125-14.ngrok-free.app/tonconnect-manifest.json')
        
   
    async def handle_crypto_payment_button(self, update: Update, context: CallbackContext) -> None:
        if update.message.text == u"\U0001F4B8 Оплатить криптой":
            # Создание клавиатуры с кнопкой для открытия Web App
            keyboard = [
                [InlineKeyboardButton("Открыть веб-приложение", web_app=WebAppInfo(url=self.web_app_url))]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text("Нажмите на кнопку ниже, чтобы открыть веб-приложение:", reply_markup=reply_markup)


    async def wait_for_wallet_connection(self):
        """
        Wait for the user to connect their wallet.
        This method should be improved with actual logic to wait for the connection.
        """
        await asyncio.sleep(10)  # Simulate waiting for the connection (replace with actual logic)
        return True

    async def request_payment(self, update: Update):
        """
        Request payment from the connected wallet.
        """
        # Create a transaction request
        transaction = {
            'valid_until': 1681223913,
            'messages': [
                {
                    'address': '0:UQCnZkgWwyit_7PDfDWM33CQBxewjgjBXGhWmYtJ3sgOQz8e',  # Replace with your TON wallet address
                    'amount': '100000000',  # Amount in nanocoins (1 TON = 10^9 nanocoins)
                }
            ]
        }

        try:
            result = await self.connector.send_transaction(transaction)
            await update.message.reply_text("Transaction was sent successfully")
        except Exception as e:
            await update.message.reply_text(f"An error occurred: {str(e)}")


   # @token_limit_check
   # @subscription_required    
    async def start(self, update: Update, context: CallbackContext) -> None:
        user_id = update.effective_user.id
        user_username = update.effective_user.username or 'анонимный пользователь'  # Получаем username или ставим заполнитель
    
        # Вступительное сообщение для пользователя
        message_text = (
            "\U0001F916Привет, Я Genius_bro_bot - бот с новейшим искусственным интеллектом и твой помощник почти по любому вопросу.\n\n"
            "\U000026A1Я могу:\n\n"
            "\U000027A1Находить любую интересующую информацию из всей глобальной сети во много раз эффективнее и быстрее, чем в Google\n"
            "\U000027A1Давать мне задачи типа написать курсовую работу, программный код, бизнес-план или просто поговорить.\n"
            "\U000027A1Рисовать любые изображения просто введя описание картинки\n\n"
            "\U000026A1Мои преимущества:\n\n"
            "\U0001F45BОчень выгодная подписка (350 рублей в месяц вместо 20-40$ при работе напрямую с ChatGPT)\n"
            "\U0001F4B3Оплата картами российских банков\n"
            "\U00002708Возможность пользоваться новейшими технологиями искусственного интеллекта не выходя из Telegram\n"
            "\U0001F5E3Я понимаю ГОЛОСОВЫЕ СООБЩЕНИЯ, тебе не нужно даже писать.\n"
            "\U0001F4B5Ты можешь зарабатывать деньги с помощью меня, приглашая друзей\n\n"
            "Запиши ГОЛОСОВОЕ сообщение или напиши текст"
        )
    
        if not db.user_exists(user_id):
            start_command = update.message.text
            referrer_id = start_command[7:]  # Извлекаем реферальный ID
    
            if referrer_id and str(referrer_id) != str(user_id):
                db.add_user(user_id, username=user_username, referrer_id=referrer_id)  # Добавляем пользователя с username
                await context.bot.send_message(chat_id=user_id, text=f"Регистрация успешна. Ваш реферальный ID: {referrer_id}")
    
                try:
                    # Сообщение для реферера с указанием username нового пользователя
                    referrer_message = f"По вашей ссылке зарегистрировался новый пользователь: @{user_username}"
                    await context.bot.send_message(referrer_id, referrer_message)
                    # Начисление баланса
                except Exception as e:
                    print(f"Ошибка при отправке сообщения: {e}")
            elif referrer_id and str(referrer_id) == str(user_id):
                await context.bot.send_message(chat_id=user_id, text="Нельзя регистрироваться по собственной реферальной ссылке.")
            else:
                db.add_user(user_id, username=user_username)
                await context.bot.send_message(chat_id=user_id, text=message_text)
        else:
            # Если пользователь уже существует в базе данных, отправляем вступительное сообщение
            await context.bot.send_message(chat_id=user_id, text=message_text)
        
        # Отправка основного меню пользователя (уже существует в вашем коде)
        keyboard = [
            [u"\U0001F46B Позвать Друга", "Assistant"],        # Эмодзи двух человек для "Позови Друга"
            [u"\U0001F4AC Помощь", u"\U0001F4B3 Подписка"],   # Эмодзи разговорного пузыря для "help" и кредитной карты для "Подписка"
            [u"\U0001F504 Сбросить диалог", u"\U0001F3A8 Создать Изображение"],  # Эмодзи стрелки для "reset" и палитры с кистью для "Нарисовать Картинку"              
            
            
                 ]
       
        
        # Добавление кнопки веб-приложения
       # keyboard.append([InlineKeyboardButton("Открыть веб-приложение", web_app="https://example.com/webapp")])
        
        # Создание объекта клавиатуры
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        # Отправка сообщения с клавиатурой
        await update.message.reply_text(message_text, reply_markup=reply_markup)
        
    @subscription_required
    async def help(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Shows the help menu.
        """
        bot_language = self.config['bot_language']
    
        # Предполагая, что у вас есть функция localized_text, которая возвращает переведенный текст
        help_descriptions = (
           "Полезная информация о боте и не только @genius_bro_channel\n\n"
           "Обсудить меня ты можешь  в нашем чате @genius_bro_chat\n\n"
           "Поддержка: @genius_bro_boss\n\n"
        )
    
        # Соединяем все описания в один текст
        
    
        # Отправляем итоговый текст пользователю
        await update.message.reply_text(help_descriptions, disable_web_page_preview=True)

    @token_limit_check
    @subscription_required  
    async def handle_image(self, update: Update, context: CallbackContext):
          # Проверяем, отправил ли пользователь фотографию
          if update.message.photo:
              # Получаем самую большую версию фото
              photo_file_id = update.message.photo[-1].file_id
      
              # Асинхронно получаем объект файла
              photo_file = await context.bot.get_file(photo_file_id)
      
              # Папка для сохранения изображений
              images_dir = '/root/geniusBro/chatgpt-telegram-bot/images'
              if not os.path.exists(images_dir):
                  os.makedirs(images_dir)
      
              # Путь, где будет сохранено фото
              photo_path = os.path.join(images_dir, f'user_photo_{update.message.from_user.id}.jpg')
      
              # Скачивание файла с использованием URL
              response = requests.get(photo_file.file_path)
              if response.status_code == 200:
                  with open(photo_path, 'wb') as file:
                      file.write(response.content)
      
                  # Сохраняем путь к изображению в user_data для использования в следующем шаге
                  context.user_data['photo_path'] = photo_path
      
                  # Запрашиваем у пользователя текстовый запрос для анализа изображения
                  await update.message.reply_text("Введите текстовый запрос для анализа изображения.")
      
                  # Переход в состояние USER_INPUT
                  return USER_INPUT
              else:
                  await update.message.reply_text("Ошибка при сохранении фотографии.")
                  return ConversationHandler.END
      
          else:
              # Если фотография не была отправлена
              await update.message.reply_text("Пожалуйста, отправьте фотографию.")
              return ConversationHandler.END
              
              
    async def analyze_image(self, update: Update, context: CallbackContext):
        # Получаем запрос пользователя
        user_query = update.message.text

        # Получаем путь к сохраненной фотографии
        photo_path = context.user_data.get('photo_path')
        if not photo_path:
            await update.message.reply_text("Произошла ошибка, пожалуйста, повторите отправку изображения.")
            return

        # Кодируем изображение в base64
        with open(photo_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Запрос к GPT-4 Vision API
        response = self.query_gpt_vision(user_query, base64_image)

        # Отправка результата анализа изображения пользователю
        if response:
            # Разделяем ответ на части, если он слишком длинный
            chunks = split_into_chunks(response)
    
            # Отправляем каждую часть как отдельное сообщение
            for chunk in chunks:
                await update.message.reply_text(chunk)
    
            return ConversationHandler.END
        else:
            await update.message.reply_text("Не удалось получить анализ изображения.")
            return ConversationHandler.END
            
            

    def query_gpt_vision(self, user_query, base64_image):
        api_key = OPENAI_API_KEY
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_query},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 1200
        }
    
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', "Не удалось интерпретировать ответ API.")
        else:
            return f"Ошибка API: {response.status_code} - {response.text}"
                 
    #@token_limit_check    
    #@subscription_required
    async def invite(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
          try:
              user_id = update.effective_user.id
              if db.user_exists(user_id):
                  # Считаем общее количество рефералов
                  referrals_count = db.count_referrals(user_id)
      
                  # Считаем количество оплаченных рефералов
                  paid_referrals_count = db.count_paid_referrals(user_id)
      
                  referral_link = f"https://t.me/genius_bro_bot?start={user_id}"
                  referral_bonus_per_user = 200
                  total_referral_bonus = paid_referrals_count * referral_bonus_per_user  # Используем количество оплаченных рефералов
      
                  stats_message = (
                      u"\u27A1\uFE0FТвоя реферальная ссылка:\n"
                      f"{referral_link}\n"
                      u"\U0001F517Поделись этой ссылкой с друзьями.\n"
                      "Когда друг оплатит подписку, ты получишь 200 рублей на баланс.\n"
                      u"Этими средствами ты можешь оплатить подписку в боте либо вывести деньги \U0001F4B5\n"
                      f"Количество всех рефералов: {referrals_count}\n"
                      f"Количество оплаченных рефералов: {paid_referrals_count}\n"
                      f"Общий баланс от оплаченных рефералов: {total_referral_bonus} руб."
                  )
      
                  await context.bot.send_message(chat_id=user_id, text=stats_message)
              else:
                  await context.bot.send_message(chat_id=user_id, text="Пользователь не найден в базе данных.")
          except Exception as e:
              logging.error(f"Ошибка в invite: {e}")
              await context.bot.send_message(chat_id=user_id, text="Произошла ошибка при обработке вашего запроса.")

          
    @token_limit_check    
    @subscription_required     
    async def tts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an speech for the given input using TTS APIs
        """
        if not self.config['enable_tts_generation'] \
                or not await self.check_allowed_and_within_budget(update, context):
            return

        tts_query = message_text(update.message)
        if tts_query == '':
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text('tts_no_prompt', self.config['bot_language'])
            )
            return

        logging.info(f'New speech generation request received from user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')

        async def _generate():
            try:
                speech_file, text_length = await self.openai.generate_speech(text=tts_query)

                await update.effective_message.reply_voice(
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    voice=speech_file
                )
                speech_file.close()
                # add image request to users usage tracker
                user_id = update.message.from_user.id
                self.usage[user_id].add_tts_request(text_length, self.config['tts_model'], self.config['tts_prices'])
                # add guest chat request to guest usage tracker
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage["guests"].add_tts_request(text_length, self.config['tts_model'], self.config['tts_prices'])

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('tts_fail', self.config['bot_language'])}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN
                )

        await wrap_with_indicator(update, context, _generate, constants.ChatAction.UPLOAD_VOICE)         
        
    @token_limit_check    
    @subscription_required
    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
            """
            Returns token usage statistics for current day and month.
            """
            if not await is_allowed(self.config, update, context):
                logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                                f'is not allowed to request their usage statistics')
                await self.send_disallowed_message(update, context)
                return
    
            logging.info(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                         f'requested their usage statistics')
    
            user_id = update.message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)
    
            tokens_today, tokens_month = self.usage[user_id].get_current_token_usage()
            images_today, images_month = self.usage[user_id].get_current_image_count()
            (transcribe_minutes_today, transcribe_seconds_today, transcribe_minutes_month,
             transcribe_seconds_month) = self.usage[user_id].get_current_transcription_duration()
            characters_today, characters_month = self.usage[user_id].get_current_tts_usage()
            current_cost = self.usage[user_id].get_current_cost()
    
            chat_id = update.effective_chat.id
            chat_messages, chat_token_length = self.openai.get_conversation_stats(chat_id)
            remaining_budget = get_remaining_budget(self.config, self.usage, update)
            bot_language = self.config['bot_language']
            
            text_current_conversation = (
                f"*{localized_text('stats_conversation', bot_language)[0]}*:\n"
                f"{chat_messages} {localized_text('stats_conversation', bot_language)[1]}\n"
                f"{chat_token_length} {localized_text('stats_conversation', bot_language)[2]}\n"
                f"----------------------------\n"
            )
            
            # Check if image generation is enabled and, if so, generate the image statistics for today
            text_today_images = ""
            if self.config.get('enable_image_generation', False):
                text_today_images = f"{images_today} {localized_text('stats_images', bot_language)}\n"
    
            text_today_tts = ""
            if self.config.get('enable_tts_generation', False):
                text_today_tts = f"{characters_today} {localized_text('stats_tts', bot_language)}\n"
            
            text_today = (
                f"*{localized_text('usage_today', bot_language)}:*\n"
                f"{tokens_today} {localized_text('stats_tokens', bot_language)}\n"
                f"{text_today_images}"  # Include the image statistics for today if applicable
                f"{text_today_tts}"
                f"{transcribe_minutes_today} {localized_text('stats_transcribe', bot_language)[0]} "
                f"{transcribe_seconds_today} {localized_text('stats_transcribe', bot_language)[1]}\n"
                f"{localized_text('stats_total', bot_language)}{current_cost['cost_today']:.2f}\n"
                f"----------------------------\n"
            )
            
            text_month_images = ""
            if self.config.get('enable_image_generation', False):
                text_month_images = f"{images_month} {localized_text('stats_images', bot_language)}\n"
    
            text_month_tts = ""
            if self.config.get('enable_tts_generation', False):
                text_month_tts = f"{characters_month} {localized_text('stats_tts', bot_language)}\n"
            
            # Check if image generation is enabled and, if so, generate the image statistics for the month
            text_month = (
                f"*{localized_text('usage_month', bot_language)}:*\n"
                f"{tokens_month} {localized_text('stats_tokens', bot_language)}\n"
                f"{text_month_images}"  # Include the image statistics for the month if applicable
                f"{text_month_tts}"
                f"{transcribe_minutes_month} {localized_text('stats_transcribe', bot_language)[0]} "
                f"{transcribe_seconds_month} {localized_text('stats_transcribe', bot_language)[1]}\n"
                f"{localized_text('stats_total', bot_language)}{current_cost['cost_month']:.2f}"
            )
    
            # text_budget filled with conditional content
            text_budget = "\n\n"
            budget_period = self.config['budget_period']
            if remaining_budget < float('inf'):
                text_budget += (
                    f"{localized_text('stats_budget', bot_language)}"
                    f"{localized_text(budget_period, bot_language)}: "
                    f"${remaining_budget:.2f}.\n"
                )
            # No longer works as of July 21st 2023, as OpenAI has removed the billing API
            # add OpenAI account information for admin request
            # if is_admin(self.config, user_id):
            #     text_budget += (
            #         f"{localized_text('stats_openai', bot_language)}"
            #         f"{self.openai.get_billing_current_month():.2f}"
            #     )
    
            usage_text = text_current_conversation + text_today + text_month + text_budget
            await update.message.reply_text(usage_text, parse_mode=constants.ParseMode.MARKDOWN)
  
          
        
    @token_limit_check    
    @subscription_required
    async def resend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resend the last request
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name}  (id: {update.message.from_user.id})'
                            f' is not allowed to resend the message')
            await self.send_disallowed_message(update, context)
            return

        chat_id = update.effective_chat.id
        if chat_id not in self.last_message:
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id})'
                            f' does not have anything to resend')
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text('resend_failed', self.config['bot_language'])
            )
            return

        # Update message text, clear self.last_message and send the request to prompt
        logging.info(f'Resending the last prompt from user: {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')
        with update.message._unfrozen() as message:
            message.text = self.last_message.pop(chat_id)

        await self.prompt(update=update, context=context)
        
        
        
    @token_limit_check    
    @subscription_required
    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resets the conversation.
        """
        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            f'is not allowed to reset the conversation')
            await self.send_disallowed_message(update, context)
            return

        logging.info(f'Resetting the conversation for user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})...')

        chat_id = update.effective_chat.id
        reset_content = message_text(update.message)
        self.openai.reset_chat_history(chat_id=chat_id, content=reset_content)
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=localized_text('reset_done', self.config['bot_language'])
        )
        
    @image_limit_check   
    @token_limit_check    
    @subscription_required
    async def image(self, update: Update, context: ContextTypes.DEFAULT_TYPE,  user_input=None):
        """
        Generates an image for the given prompt using DALL·E APIs
        """
        if not self.config['enable_image_generation'] \
                or not await self.check_allowed_and_within_budget(update, context):
            return
        if user_input:
            image_query = user_input
        else:
            image_query = message_text(update.message)
        
            if image_query == '':
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=localized_text('image_no_prompt', self.config['bot_language'])
                )
                return
    
            logging.info(f'New image generation request received from user {update.message.from_user.name} '
                         f'(id: {update.message.from_user.id})')
             # Отправляем уведомление пользователю о загрузке фото
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=telegram.ChatAction.UPLOAD_PHOTO)
            
            
        async def _generate():
            try:
                image_url, image_size = await self.openai.generate_image(prompt=image_query)
                
                print("image_url:", image_url, image_size )  # Или используйте logging для более структурированного подхода
      
                await update.effective_message.reply_photo(
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    photo=image_url
                )
                # add image request to users usage tracker
                user_id = update.message.from_user.id
                self.usage[user_id].add_image_request(image_size, self.config['image_prices'])
                # add guest chat request to guest usage tracker
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage["guests"].add_image_request(image_size, self.config['image_prices'])

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                  #  text=f"{localized_text('image_fail', self.config['bot_language'])}: {str(e)}",
                    text=f"{localized_text('image_fail', self.config['bot_language'])}",
                    parse_mode=constants.ParseMode.MARKDOWN
                )

        await wrap_with_indicator(update, context, _generate, constants.ChatAction.UPLOAD_PHOTO)
    
    

    @token_limit_check    
    @subscription_required
    async def transcribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Transcribe audio messages.
        """
        if not self.config['enable_transcription'] or not await self.check_allowed_and_within_budget(update, context):
            return

        if is_group_chat(update) and self.config['ignore_group_transcriptions']:
            logging.info(f'Transcription coming from group chat, ignoring...')
            return

        chat_id = update.effective_chat.id
        filename = update.message.effective_attachment.file_unique_id

        async def _execute():
            filename_mp3 = f'{filename}.mp3'
            bot_language = self.config['bot_language']
            try:
                media_file = await context.bot.get_file(update.message.effective_attachment.file_id)
                await media_file.download_to_drive(filename)
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=(
                        f"{localized_text('media_download_fail', bot_language)[0]}: "
                        f"{str(e)}. {localized_text('media_download_fail', bot_language)[1]}"
                    ),
                    parse_mode=constants.ParseMode.MARKDOWN
                )
                return

            try:
                audio_track = AudioSegment.from_file(filename)
                audio_track.export(filename_mp3, format="mp3")
                logging.info(f'New transcribe request received from user {update.message.from_user.name} '
                             f'(id: {update.message.from_user.id})')

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=localized_text('media_type_fail', bot_language)
                )
                if os.path.exists(filename):
                    os.remove(filename)
                return

            user_id = update.message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

            try:
                transcript = await self.openai.transcribe(filename_mp3)

                transcription_price = self.config['transcription_price']
                self.usage[user_id].add_transcription_seconds(audio_track.duration_seconds, transcription_price)

                allowed_user_ids = self.config['allowed_user_ids'].split(',')
                if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                    self.usage["guests"].add_transcription_seconds(audio_track.duration_seconds, transcription_price)

                # check if transcript starts with any of the prefixes
                response_to_transcription = any(transcript.lower().startswith(prefix.lower()) if prefix else False
                                                for prefix in self.config['voice_reply_prompts'])

                if self.config['voice_reply_transcript'] and not response_to_transcription:

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    transcript_output = f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\""
                    chunks = split_into_chunks(transcript_output)

                    for index, transcript_chunk in enumerate(chunks):
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update) if index == 0 else None,
                            text=transcript_chunk,
                            parse_mode=constants.ParseMode.MARKDOWN
                        )
                else:
                    # Get the response of the transcript
                    response, total_tokens = await self.openai.get_chat_response(chat_id=chat_id, query=transcript)

                    self.usage[user_id].add_chat_tokens(total_tokens, self.config['token_price'])
                    if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                        self.usage["guests"].add_chat_tokens(total_tokens, self.config['token_price'])

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    transcript_output = (
                        f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\"\n\n"
                        f"_{localized_text('answer', bot_language)}:_\n{response}"
                    )
                    chunks = split_into_chunks(transcript_output)

                    for index, transcript_chunk in enumerate(chunks):
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update) if index == 0 else None,
                            text=transcript_chunk,
                            parse_mode=constants.ParseMode.MARKDOWN
                        )

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('transcribe_fail', bot_language)}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN
                )
            finally:
                if os.path.exists(filename_mp3):
                    os.remove(filename_mp3)
                if os.path.exists(filename):
                    os.remove(filename)

        await wrap_with_indicator(update, context, _execute, constants.ChatAction.TYPING)
        

        
    @token_limit_check    
    @subscription_required
    async def go_back(self, update: Update, context: CallbackContext) -> None:
        # Создаем кнопки основного меню
       
        
        keyboard = [
            [u"\U0001F46B Позвать Друга"],        # Эмодзи двух человек для "Позови Друга"
            [u"\U0001F4AC Помощь", u"\U0001F4B3 Подписка"],   # Эмодзи разговорного пузыря для "help" и кредитной карты для "Подписка"
            [u"\U0001F504 Сбросить диалог", u"\U0001F3A8 Создать Изображение"]  # Эмодзи стрелки для "reset" и палитры с кистью для "Нарисовать Картинку"
                    ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
        # Отправляем сообщение с основным меню
        await update.message.reply_text("Выберите опцию:", reply_markup=reply_markup)
        
        
        
    @subscription_required
    async def go_back2(self, update: Update, context: CallbackContext) -> None:
        # Создаем кнопки основного меню
       
        
        keyboard = [
            [u"\U0001F46B Позвать Друга"],        # Эмодзи двух человек для "Позови Друга"
            [u"\U0001F4AC Помощь", u"\U0001F4B3 Подписка"],   # Эмодзи разговорного пузыря для "help" и кредитной карты для "Подписка"
            [u"\U0001F504 Сбросить диалог", u"\U0001F3A8 Создать Изображение"]  # Эмодзи стрелки для "reset" и палитры с кистью для "Нарисовать Картинку"
                    ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
        # Отправляем сообщение с основным меню
        await update.message.reply_text("Выберите опцию:", reply_markup=reply_markup)    
        
    @token_limit_check    
    @subscription_required
    async def start_image_input(self, update: Update, context: CallbackContext) -> int:
        # Отправляем сообщение с просьбой ввести запрос
        await update.message.reply_text("Введите ваш запрос:")
        return DRAW_IMAGE
        
    @token_limit_check    
    @subscription_required
    async def start_image_input__voice(self, update: Update, context: CallbackContext) -> int:
        # Отправляем сообщение с просьбой ввести запрос или отправить голосовое сообщение
        await update.message.reply_text("Введите ваш запрос или отправьте голосовое сообщение:")
        return VOICE_INPUT    
        
    @token_limit_check
    @subscription_required
    async def handle_user_input(self, update: Update, context: CallbackContext) -> int:
    
        user_input = context.user_data.get('input_text') or update.message.text
        # Добавляем префикс '/image' к пользовательскому вводу
        image_command_input = '/image ' + user_input
        # Обрабатываем модифицированный ввод пользователя
        await self.image(update, context, user_input=image_command_input)
        return ConversationHandler.END
        
    
    async def cancel(self, update: Update, context: CallbackContext) -> int:
        await update.message.reply_text('Операция отменена.')
        return ConversationHandler.END   
     
  
    async def platej(self, update: Update, context: CallbackContext) -> None:
          keyboard = [
                        [u"\U0001F4B3 Оплатить картой", u"\U0001F4B8 Оплатить криптой"],
                        [u"\U0001F6AB Отменить подписку", u"\u2B05 назад"]
                        
                    ]
                      
          reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
          text = (
                      "Данная подписка дает возможность пользоваться ботом:\n"
                      "\u25AA\uFE0FДелать запрос голосовым сообщением или текстом\n"
                      "\u25AA\uFE0FДелать запрос картинки\n"
                      "\u25AA\uFE0FСкачивать аудио с Youtube\n"
                      "\u25AA\uFE0FВ разработке много новых функций\n\n"
                      "Оформить подписку\n"
                      "\u27A1\uFE0F На месяц - 350 RUB\n"
                      "\u27A1\uFE0F На три месяца - 1000 RUB\n\n"
                      "Оплата\n"
                      "\U0001F4B3: VISA/MC/МИР\n"
                      "\U0001F4B5 Доступна оплата криптовалютой (USDT)"
                  )

            
          await update.message.reply_text(text, reply_markup=reply_markup)
          
    async def ask_cancel_subscription(self, update: Update, context: CallbackContext):
          keyboard = [
              [InlineKeyboardButton("Да", callback_data="confirm_cancel_subscription"),
               InlineKeyboardButton("Нет", callback_data="decline_cancel_subscription")]
          ]
          reply_markup = InlineKeyboardMarkup(keyboard)
          await update.message.reply_text(
              "Вы уверены, что хотите отменить подписку?", reply_markup=reply_markup
          )      
          
    async def handle_subscription_decision(self, update: Update, context: CallbackContext):
          query = update.callback_query
          await query.answer()
      
          user_id = query.from_user.id
          if query.data == "confirm_cancel_subscription":
              db.cancel_subscription(user_id)
              await query.edit_message_text("Ваша подписка успешно отменена.")
              # Дополнительная логика для удаления платежной информации, если это необходимо
          elif query.data == "decline_cancel_subscription":
              # Создаем клавиатуру для возврата в предыдущее меню
              keyboard = [
                        [u"\U0001F4B3 Оплатить картой", u"\U0001F4B8 Оплатить криптой"],
                        [u"\U0001F6AB Отменить подписку", u"\u2B05 назад"]
                        
                    ]
              
              reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
              await context.bot.send_message(chat_id=query.message.chat_id, 
                                       text="Выберите опцию:", 
                                       reply_markup=reply_markup)
        #    await update.message.reply_text("Ваша подписка успешно отменена и платежный метод удален из нашей системы.")

    
    
    async def handle_payment(self, update: Update, context: CallbackContext) -> None:
          user_id = None
          text = (
                    "Пожалуйста, введите ваш email для продолжения:\n\n"
                    "\U0001F4E7 это нужно для того, чтобы отправить вам чек.\n"
                    "\U0001F4B3 Этого требует законодательство."
                )

      
          # Проверяем тип обновления
          if update.callback_query:
              query = update.callback_query
              await query.answer()
              user_id = query.from_user.id
          elif update.message:
              user_id = update.message.from_user.id
      
          # Проверяем, что user_id был получен
          if user_id is not None:
              await context.bot.send_message(chat_id=user_id, text=text)
      
              # Сохраняем флаг, указывающий, что ожидаем ввода email
              context.user_data['awaiting_email'] = True
      
              # Возвращаем состояние, указывающее на ожидание ввода email
              return EMAIL
          else:
              # В случае ошибки
              await update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте ещё раз.")
              return ConversationHandler.END


   
        
            
    async def email_response(self, update: Update, context: CallbackContext) -> int:
        user_id = update.effective_user.id
        email = update.message.text
        
    
        # Логирование полученного email
        logger.info(f"Received email: {email} from user: {user_id}")
        def is_valid_email(email):
          pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
          return re.match(pattern, email) is not None
        
        if not is_valid_email(email):
            await update.message.reply_text("Некорректный формат электронной почты. Пожалуйста, введите действительный адрес.")
            return ConversationHandler.END
    
        try:
            # Сохраняем email в базе данных
            db.save_user_email(user_id, email)
            logger.info("Email saved in database")
        except Exception as e:
            logger.error(f"Error saving email to database: {e}")
            await update.message.reply_text("Ошибка при сохранении электронной почты. Пожалуйста, попробуйте позже.")
            return ConversationHandler.END
    
        # Продолжаем процесс создания платежа
        amount = 350  # Сумма подписки
        return_url = "https://example.com/return_url"
    
        try:
            # Создание платежа в YooKassa с использованием email
            payment = create_and_save_payment_method(amount, return_url, email)
            logger.info("Payment created with YooKassa")
        except Exception as e:
            logger.error(f"Error creating payment with YooKassa: {e}")
            await update.message.reply_text("Ошибка при создании платежа. Пожалуйста, попробуйте позже.")
            return ConversationHandler.END
    
        if payment.confirmation and payment.confirmation.type == 'redirect':
            # Отправка пользователя на страницу оплаты
            conf_url = payment.confirmation.confirmation_url
            await update.message.reply_text(
                text="Перейдите по ссылке для оплаты подписки:",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Оплатить", url=conf_url)]
                ])
            )
            # Запуск асинхронной задачи для проверки статуса платежа
            asyncio.create_task(check_payment_status(payment.id, context, user_id))
        else:
            await update.message.reply_text("Ошибка при создании платежа. Пожалуйста, попробуйте позже.")
    
        return ConversationHandler.END
        
    
    async def cancel(update: Update, context: CallbackContext) -> int:
        user_id = update.effective_user.id
        await context.bot.send_message(chat_id=user_id, text="Оплата отменена.")
        return ConversationHandler.END
        
        
        
        
     
    @token_limit_check   
    @subscription_required
    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        React to incoming messages and respond accordingly.
        """
        if update.edited_message or not update.message or update.message.via_bot:
            return

        if not await self.check_allowed_and_within_budget(update, context):
            return
            
        
        if update.message.text.lower() == 'stats':
           await self.stats(update, context)
           return
      #  if update.message.text == u"\U0001F3A8 Нарисовать Картинку":
      #     await self.image_one(update, context)
      #     return
        if update.message.text == u"\U0001F504 Сбросить диалог":
           await self.reset(update, context)
           return
        
       # if update.message.text == "Открыть веб-приложение":
           # Ничего не делать при нажатии на "Открыть веб-приложение"
      #     return      
           
           
        logging.info(
            f'New message received from user {update.message.from_user.name} (id: {update.message.from_user.id})')
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        prompt = message_text(update.message)
        self.last_message[chat_id] = prompt

        if is_group_chat(update):
            trigger_keyword = self.config['group_trigger_keyword']

            if prompt.lower().startswith(trigger_keyword.lower()) or update.message.text.lower().startswith('/chat'):
                if prompt.lower().startswith(trigger_keyword.lower()):
                    prompt = prompt[len(trigger_keyword):].strip()

                if update.message.reply_to_message and \
                        update.message.reply_to_message.text and \
                        update.message.reply_to_message.from_user.id != context.bot.id:
                    prompt = f'"{update.message.reply_to_message.text}" {prompt}'
            else:
                if update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id:
                    logging.info('Message is a reply to the bot, allowing...')
                else:
                    logging.warning('Message does not start with trigger keyword, ignoring...')
                    return

        try:
            total_tokens = 0

            if self.config['stream']:
                await update.effective_message.reply_chat_action(
                    action=constants.ChatAction.TYPING,
                    message_thread_id=get_thread_id(update)
                )

                stream_response = self.openai.get_chat_response_stream(chat_id=chat_id, query=prompt)
                i = 0
                prev = ''
                sent_message = None
                backoff = 0
                stream_chunk = 0

                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        return await handle_direct_result(self.config, update, content)

                    if len(content.strip()) == 0:
                        continue

                    stream_chunks = split_into_chunks(content)
                    if len(stream_chunks) > 1:
                        content = stream_chunks[-1]
                        if stream_chunk != len(stream_chunks) - 1:
                            stream_chunk += 1
                            try:
                                await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                              stream_chunks[-2])
                            except:
                                pass
                            try:
                                sent_message = await update.effective_message.reply_text(
                                    message_thread_id=get_thread_id(update),
                                    text=content if len(content) > 0 else "..."
                                )
                            except:
                                pass
                            continue

                    cutoff = get_stream_cutoff_values(update, content)
                    cutoff += backoff

                    if i == 0:
                        try:
                            if sent_message is not None:
                                await context.bot.delete_message(chat_id=sent_message.chat_id,
                                                                 message_id=sent_message.message_id)
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config, update),
                                text=content,
                            )
                        except:
                            continue

                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content

                        try:
                            use_markdown = tokens != 'not_finished'
                            await edit_message_with_retry(context, chat_id, str(sent_message.message_id),
                                                          text=content, markdown=use_markdown)

                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue

                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue

                        except Exception:
                            backoff += 5
                            continue

                        await asyncio.sleep(0.01)

                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)

            else:
                async def _reply():
                    nonlocal total_tokens
                    response, total_tokens = await self.openai.get_chat_response(chat_id=chat_id, query=prompt)

                    if is_direct_result(response):
                        return await handle_direct_result(self.config, update, response)

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    chunks = split_into_chunks(response)

                    for index, chunk in enumerate(chunks):
                        try:
                            await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(self.config,
                                                                            update) if index == 0 else None,
                                text=chunk,
                                parse_mode=constants.ParseMode.MARKDOWN
                            )
                        except Exception:
                            try:
                                await update.effective_message.reply_text(
                                    message_thread_id=get_thread_id(update),
                                    reply_to_message_id=get_reply_to_message_id(self.config,
                                                                                update) if index == 0 else None,
                                    text=chunk
                                )
                            except Exception as exception:
                                raise exception

                await wrap_with_indicator(update, context, _reply, constants.ChatAction.TYPING)

            add_chat_request_to_usage_tracker(self.usage, self.config, user_id, total_tokens)

        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f"{localized_text('chat_fail', self.config['bot_language'])} {str(e)}",
                parse_mode=constants.ParseMode.MARKDOWN
            )
    @token_limit_check        
    @subscription_required
    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the inline query. This is run when you type: @botusername <query>
        """
        query = update.inline_query.query
        if len(query) < 3:
            return
        if not await self.check_allowed_and_within_budget(update, context, is_inline=True):
            return

        callback_data_suffix = "gpt:"
        result_id = str(uuid4())
        self.inline_queries_cache[result_id] = query
        callback_data = f'{callback_data_suffix}{result_id}'

        await self.send_inline_query_result(update, result_id, message_content=query, callback_data=callback_data)

    async def send_inline_query_result(self, update: Update, result_id, message_content, callback_data=""):
        """
        Send inline query result
        """
        try:
            reply_markup = None
            bot_language = self.config['bot_language']
            if callback_data:
                reply_markup = InlineKeyboardMarkup([[
                    InlineKeyboardButton(text=f'{localized_text("answer_with_chatgpt", bot_language)}',
                                         callback_data=callback_data)
                ]])

            inline_query_result = InlineQueryResultArticle(
                id=result_id,
                title=localized_text("ask_chatgpt", bot_language),
                input_message_content=InputTextMessageContent(message_content),
                description=message_content,
                thumb_url='https://user-images.githubusercontent.com/11541888/223106202-7576ff11-2c8e-408d-94ea'
                          '-b02a7a32149a.png',
                reply_markup=reply_markup
            )

            await update.inline_query.answer([inline_query_result], cache_time=0)
        except Exception as e:
            logging.error(f'An error occurred while generating the result card for inline query {e}')
            
            
            
            
    @token_limit_check        
    @subscription_required
    async def handle_callback_inline_query(self, update: Update, context: CallbackContext):
        """
        Handle the callback query from the inline query result
        """
        callback_data = update.callback_query.data
        user_id = update.callback_query.from_user.id
        inline_message_id = update.callback_query.inline_message_id
        name = update.callback_query.from_user.name
        callback_data_suffix = "gpt:"
        query = ""
        bot_language = self.config['bot_language']
        answer_tr = localized_text("answer", bot_language)
        loading_tr = localized_text("loading", bot_language)

        try:
            if callback_data.startswith(callback_data_suffix):
                unique_id = callback_data.split(':')[1]
                total_tokens = 0

                # Retrieve the prompt from the cache
                query = self.inline_queries_cache.get(unique_id)
                if query:
                    self.inline_queries_cache.pop(unique_id)
                else:
                    error_message = (
                        f'{localized_text("error", bot_language)}. '
                        f'{localized_text("try_again", bot_language)}'
                    )
                    await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                                  text=f'{query}\n\n_{answer_tr}:_\n{error_message}',
                                                  is_inline=True)
                    return

                unavailable_message = localized_text("function_unavailable_in_inline_mode", bot_language)
                if self.config['stream']:
                    stream_response = self.openai.get_chat_response_stream(chat_id=user_id, query=query)
                    i = 0
                    prev = ''
                    backoff = 0
                    async for content, tokens in stream_response:
                        if is_direct_result(content):
                            cleanup_intermediate_files(content)
                            await edit_message_with_retry(context, chat_id=None,
                                                          message_id=inline_message_id,
                                                          text=f'{query}\n\n_{answer_tr}:_\n{unavailable_message}',
                                                          is_inline=True)
                            return

                        if len(content.strip()) == 0:
                            continue

                        cutoff = get_stream_cutoff_values(update, content)
                        cutoff += backoff

                        if i == 0:
                            try:
                                await edit_message_with_retry(context, chat_id=None,
                                                              message_id=inline_message_id,
                                                              text=f'{query}\n\n{answer_tr}:\n{content}',
                                                              is_inline=True)
                            except:
                                continue

                        elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                            prev = content
                            try:
                                use_markdown = tokens != 'not_finished'
                                divider = '_' if use_markdown else ''
                                text = f'{query}\n\n{divider}{answer_tr}:{divider}\n{content}'

                                # We only want to send the first 4096 characters. No chunking allowed in inline mode.
                                text = text[:4096]

                                await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                                              text=text, markdown=use_markdown, is_inline=True)

                            except RetryAfter as e:
                                backoff += 5
                                await asyncio.sleep(e.retry_after)
                                continue
                            except TimedOut:
                                backoff += 5
                                await asyncio.sleep(0.5)
                                continue
                            except Exception:
                                backoff += 5
                                continue

                            await asyncio.sleep(0.01)

                        i += 1
                        if tokens != 'not_finished':
                            total_tokens = int(tokens)

                else:
                    async def _send_inline_query_response():
                        nonlocal total_tokens
                        # Edit the current message to indicate that the answer is being processed
                        await context.bot.edit_message_text(inline_message_id=inline_message_id,
                                                            text=f'{query}\n\n_{answer_tr}:_\n{loading_tr}',
                                                            parse_mode=constants.ParseMode.MARKDOWN)

                        logging.info(f'Generating response for inline query by {name}')
                        response, total_tokens = await self.openai.get_chat_response(chat_id=user_id, query=query)

                        if is_direct_result(response):
                            cleanup_intermediate_files(response)
                            await edit_message_with_retry(context, chat_id=None,
                                                          message_id=inline_message_id,
                                                          text=f'{query}\n\n_{answer_tr}:_\n{unavailable_message}',
                                                          is_inline=True)
                            return

                        text_content = f'{query}\n\n_{answer_tr}:_\n{response}'

                        # We only want to send the first 4096 characters. No chunking allowed in inline mode.
                        text_content = text_content[:4096]

                        # Edit the original message with the generated content
                        await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                                      text=text_content, is_inline=True)

                    await wrap_with_indicator(update, context, _send_inline_query_response,
                                              constants.ChatAction.TYPING, is_inline=True)

                add_chat_request_to_usage_tracker(self.usage, self.config, user_id, total_tokens)

        except Exception as e:
            logging.error(f'Failed to respond to an inline query via button callback: {e}')
            logging.exception(e)
            localized_answer = localized_text('chat_fail', self.config['bot_language'])
            await edit_message_with_retry(context, chat_id=None, message_id=inline_message_id,
                                          text=f"{query}\n\n_{answer_tr}:_\n{localized_answer} {str(e)}",
                                          is_inline=True)

    async def check_allowed_and_within_budget(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                              is_inline=False) -> bool:
        """
        Checks if the user is allowed to use the bot and if they are within their budget
        :param update: Telegram update object
        :param context: Telegram context object
        :param is_inline: Boolean flag for inline queries
        :return: Boolean indicating if the user is allowed to use the bot
        """
        name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
        user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id

        if not await is_allowed(self.config, update, context, is_inline=is_inline):
            logging.warning(f'User {name} (id: {user_id}) is not allowed to use the bot')
            await self.send_disallowed_message(update, context, is_inline)
            return False
        if not is_within_budget(self.config, self.usage, update, is_inline=is_inline):
            logging.warning(f'User {name} (id: {user_id}) reached their usage limit')
            await self.send_budget_reached_message(update, context, is_inline)
            return False

        return True

    async def send_disallowed_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False):
        """
        Sends the disallowed message to the user.
        """
        if not is_inline:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=self.disallowed_message,
                disable_web_page_preview=True
            )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(update, result_id, message_content=self.disallowed_message)

    async def send_budget_reached_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False):
        """
        Sends the budget reached message to the user.
        """
        if not is_inline:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=self.budget_limit_message
            )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(update, result_id, message_content=self.budget_limit_message)

    async def post_init(self, application: Application) -> None:
        """
        Post initialization hook for the bot.
        """
        await application.bot.set_my_commands(self.group_commands, scope=BotCommandScopeAllGroupChats())
        await application.bot.set_my_commands(self.commands)
        
    def setup_openai_assistant(file_path):
        file = client.files.create(
            file=open(file_path, 'rb'),
            purpose="assistants"
        )
        
        assistant = client.beta.assistants.create(
        name="PDF Helper",
        instructions="You are my assistant who can answer questions from the given pdf",
        tools=[{"type": "retrieval"}],
        model="gpt-4-1106-preview",
        file_ids=[file.id]
         )
    
        return assistant
    
    
    def ask_assistant(assistant_id, file_id, question):
        """
        Sends a question to the OpenAI Assistant and retrieves the response.
    
        Parameters:
        - assistant_id: The unique identifier of the assistant
        - file_id: The file ID of the uploaded PDF
        - question: The user's question about the PDF content
    
        Returns:
        - answer: The response from the assistant
        """
        from openai import OpenAI
        client = OpenAI(api_key="your_openai_api_key")
    
        # Create a thread to manage conversation with the assistant
        thread = client.beta.threads.create()
    
        # Send the question to the assistant within the thread
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=question
        )
    
        # Start processing the question with the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )
    
        # Wait for the assistant to process the question and provide a response
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status == 'completed':
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                break
            asyncio.sleep(2)  # Use asyncio.sleep to avoid blocking in an async function
    
        # Extract the assistant's response from the messages
        for message in reversed(messages.data):
            if message.role == 'assistant':
                return message.content[0].text.value
    
        return "Sorry, I couldn't process your request. Please try again later."
    
    
    
    
    async def handle_assistant_start(self, update: Update, context: CallbackContext):
        if update.message:
            await update.message.reply_text('Пожалуйста, загрузите PDF-файл.')
        return UPLOAD_PDF




    
    
    


    def handle_pdf_upload(self, update: Update, context: CallbackContext):
        file = context.bot.getFile(update.message.document.file_id)
        file_path = f"{file.file_id}.pdf"
        file.download(file_path)
        # Set up the assistant with the uploaded PDF
        assistant = setup_openai_assistant(file_path)
        context.user_data['assistant'] = assistant
        update.message.reply_text('PDF uploaded successfully. You can now ask questions about its content.')
        return ANALYZE_PDF

    def handle_question(self, update: Update, context: CallbackContext):
        question = update.message.text
        assistant = context.user_data.get('assistant')
        if assistant:
            answer = self.openai.ask_assistant(assistant, question)
            update.message.reply_text(answer)
        else:
            update.message.reply_text('Please upload a PDF file first.')
        return ConversationHandler.END
        
        
    def cancel(self, update: Update, context: CallbackContext):
        update.message.reply_text('Operation canceled.')
        return ConversationHandler.END


            
    def run(self):
        """
        Runs the bot indefinitely until the user presses Ctrl+C
        """
        application = ApplicationBuilder() \
            .token(self.config['token']) \
            .proxy_url(self.config['proxy']) \
            .get_updates_proxy_url(self.config['proxy']) \
            .post_init(self.post_init) \
            .concurrent_updates(True) \
            .build()
            
            

        
        conv_handler = ConversationHandler(
            entry_points=[
                MessageHandler(filters.PHOTO, self.handle_image),  # For handling photos
                MessageHandler(filters.Regex(u"\U0001F3A8 Создать Изображение"), self.start_image_input),  # For "Draw Image" command
                MessageHandler(filters.Regex(u"\U0001F4B3 Оплатить картой"), self.handle_payment),  # For payment requests
                CallbackQueryHandler(self.handle_payment, pattern='^pay_subscription$'),
                MessageHandler(filters.Document.MimeType("application/pdf"), self.handle_pdf_upload),  # For uploading PDF
                
            ],
            states={
                PHOTO: [MessageHandler(filters.PHOTO, self.handle_image)],  # State for processing photos
                USER_INPUT: [MessageHandler(filters.TEXT, self.analyze_image)],  # State for text after photo
                DRAW_IMAGE: [MessageHandler(filters.TEXT, self.handle_user_input)],  # State for text after "Draw Image" command
                EMAIL: [MessageHandler(filters.TEXT , self.email_response)],  # State for email input
                UPLOAD_PDF: [MessageHandler(filters.Document.MimeType("application/pdf"), self.handle_pdf_upload)],  # State for uploading PDF
                ANALYZE_PDF: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_question)]  # State for analyzing PDF content
            },
            fallbacks=[CommandHandler('cancel', self.cancel)]  # Fallback command
        )
              
                
       
          
        
        
        application.add_handler(MessageHandler(filters.Regex("Assistant"), self.handle_assistant_start))    
        application.add_handler(MessageHandler(filters.Regex(u"\U0001F4B8 Оплатить криптой"), self.handle_crypto_payment_button))  
        application.add_handler(CallbackQueryHandler(self.handle_subscription_decision, pattern='^confirm_cancel_subscription$|^decline_cancel_subscription$'))  
        application.add_handler(MessageHandler(filters.Regex(u"\U0001F6AB Отменить подписку"), self.ask_cancel_subscription))
        application.add_handler(MessageHandler(filters.Regex(u"\U0001F46B Позвать Друга"), self.invite)) 
        application.add_handler(MessageHandler(filters.Regex(u"\U0001F4B3 Подписка"), self.platej))
        application.add_handler(MessageHandler(filters.Regex(u"\U0001F4AC Помощь"), self.help))
 
        application.add_handler(MessageHandler(filters.Regex(u"\u2B05 назад"), self.go_back2))
        application.add_handler(MessageHandler(filters.Regex('^Назад$'), self.go_back))  
        application.add_handler(conv_handler)    
        application.add_handler(CommandHandler('start', self.start))
        application.add_handler(CommandHandler('tts', self.tts))
        application.add_handler(CommandHandler('reset', self.reset))
        application.add_handler(CommandHandler('help', self.help))
        application.add_handler(CommandHandler('image', self.image))
       # application.add_handler(CommandHandler('start', self.help))
        application.add_handler(CommandHandler('stats', self.stats))
        application.add_handler(CommandHandler('resend', self.resend))
   #####     application.add_handler(MessageHandler(filters.PHOTO, self.analyze_image))
        application.add_handler(CommandHandler(
            'chat', self.prompt, filters=filters.ChatType.GROUP | filters.ChatType.SUPERGROUP)
        )
        application.add_handler(MessageHandler(
            filters.AUDIO | filters.VOICE | filters.Document.AUDIO |
            filters.VIDEO | filters.VIDEO_NOTE | filters.Document.VIDEO,
            self.transcribe))
        
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt))
        
        application.add_handler(InlineQueryHandler(self.inline_query, chat_types=[
            constants.ChatType.GROUP, constants.ChatType.SUPERGROUP, constants.ChatType.PRIVATE
        ]))
        application.add_handler(CallbackQueryHandler(self.handle_callback_inline_query))

        application.add_error_handler(error_handler)

        application.run_polling()