from telegram.ext import Updater, CommandHandler
import sqlite3 as sq
from threading import Thread

bot_token = "800792656:AAF3UcFpElvjeG3q3b-Q9JjRVSEn_c_Y6JE"


class TelegramUsersDataBase:
    def __init__(self):
        self.telegram_database = sq.connect('telegram_users.db')
        self.cursor = self.telegram_database.cursor()

    def create_table(self):
        try:
            self.cursor.execute("""CREATE TABLE telegram_user(first_name TEXT DEFAULT NULL ,
                                        last_name TEXT DEFAULT NULL,username TEXT NOT NULL ,
                                        chat_id INTEGER NOT NULL UNIQUE,
                                        user_referrer INTEGER DEFAULT NULL )""")
            self.telegram_database.commit()
            return
        except sq.OperationalError:
            return

    def add_new_user(self, *args):
        self.cursor.execute("""INSERT INTO telegram_user VALUES (?, ?, ?, ?, ?)""", args)
        self.telegram_database.commit()

    def add_referral(self, referrer_id, referral_id):
        self.cursor.execute("""UPDATE telegram_user SET user_referrer = ? WHERE user_referrer == '' AND
                                chat_id = ?""", (referrer_id, referral_id))
        self.telegram_database.commit()

    def select_referrals(self, referral_id):
        users_id = self.cursor.execute("""SELECT chat_id FROM telegram_user WHERE user_referrer == ?""",
                                       (referral_id,))
        return users_id.fetchall()

    @staticmethod
    def check_user_exist(chat_id):
        cursor = sq.connect('telegram_users.db').cursor()
        user_exist = cursor.execute("""SELECT count(*) FROM telegram_user
                                    WHERE chat_id = ?""", (chat_id,))
        return user_exist.fetchone()[0]

CHOICE, NEW_TIME = range(2)


class TelegramBot:
    def __init__(self):
        self.updater = Updater(token=bot_token)
        self.bot_users_database = TelegramUsersDataBase
        self.bot_users_database().create_table()
        self.bot_url = 'https://t.me/Fire_DetectorBot'
        self.updater.dispatcher.add_handler(CommandHandler('start', self.new_chat_user))
        self.updater.dispatcher.add_handler(CommandHandler('help', self.help_message))
        self.updater.dispatcher.add_handler(CommandHandler('check', self.send_check_photo))
        self.updater.dispatcher.add_handler(CommandHandler('settings', self.user_settings))
        self.updater.dispatcher.add_handler(CommandHandler('my_referral_link', self.get_referral_link))

    @staticmethod
    def get_bot_commands():
        return {'/start': 'Begin your chat', '/my_referral_link': 'Get referral link',
                '/check': 'Take a photo at the moment', '/help': 'List of bot commands'}

    def start_bot(self):
        self.updater.start_polling()

    def help_message(self, bot, update):
        chat_id = update.message.chat_id
        command = self.get_bot_commands()
        help_message = ''

        for key in command.keys():
            help_message += "{0} - {1}\n".format(key, command[key])
        bot.sendMessage(chat_id=chat_id, text=help_message)

    def new_chat_user(self, bot, update):
        chat_id = update.message.chat_id
        message = update.message.text.split('/start ', 1)
        if self.bot_users_database.check_user_exist(chat_id):
            bot.sendMessage(chat_id=chat_id, text="{0}, you're already start this bot".
                            format(update.message.from_user.first_name))
        else:
            try:
                self.bot_users_database().add_new_user(update.message.from_user.first_name,
                                                       update.message.from_user.last_name,
                                                       update.message.from_user.username, chat_id, '')
                bot.sendMessage(chat_id=chat_id, text="Hello, {0}".format(update.message.from_user.first_name))
            except sq.ProgrammingError:
                bot.sendMessage(chat_id=chat_id, text="Please, try later")
        if len(message) == 2:
            self.bot_users_database().add_referral(message[1], chat_id)

    def send_check_photo(self, bot, update):
        chat_id = update.message.chat_id
        photo = open('/FireDetector/check_photo/check.png', 'rb')
        bot.sendPhoto(chat_id=chat_id, photo=photo)
        referrals_id = self.bot_users_database().select_referrals(referral_id=chat_id)
        if len(referrals_id) > 0:
            for chat_id in referrals_id:
                Thread(target=self.send_to_referrals, args=(bot, chat_id[0])).start()

    @staticmethod
    def send_to_referrals(bot, chat_id):
        photo = open('/FireDetector/check_photo/check.png', 'rb')
        bot.sendPhoto(chat_id=chat_id, photo=photo)
        photo.close()

    def user_settings(self, bot, update):
        pass

    def get_referral_link(self, bot, update):
        chat_id = update.message.chat_id
        referral_link = self.bot_url + '?start=' + str(chat_id)
        bot.sendMessage(chat_id=chat_id, text=referral_link)


if __name__ == '__main__':

    x = TelegramBot()
    x.start_bot()
