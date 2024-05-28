import sqlite3

class Database:
    def __init__(self, db_file):
        self.db_file = db_file
        self.ensure_table_exists()

    def ensure_table_exists(self):
        create_table_query = """
                  CREATE TABLE IF NOT EXISTS user_wallets (
                      id INTEGER PRIMARY KEY,
                      user_id TEXT NOT NULL UNIQUE,
                      username TEXT,
                      referrer_id TEXT,
                      paid_subscription BOOLEAN NOT NULL DEFAULT 0,
                      purchased_tokens INTEGER NOT NULL DEFAULT 0,
                      payment_method_id TEXT,
                      email TEXT  -- новый столбец для хранения email
                  );
                  """
        with sqlite3.connect(self.db_file) as conn:
            conn.execute(create_table_query)
            conn.commit()

    def save_user_email(self, user_id, email):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE user_wallets SET email = ? WHERE user_id = ?", (email, user_id))
            conn.commit()

    def get_user_email(self, user_id):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            result = cursor.execute("SELECT email FROM user_wallets WHERE user_id = ?", (user_id,)).fetchone()
            return result[0] if result else None

    def user_exists(self, user_id):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            result = cursor.execute("SELECT * FROM user_wallets WHERE user_id=?", (user_id,)).fetchall()
            return bool(len(result))

    def add_user(self, user_id, username=None, referrer_id=None):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            if referrer_id is not None:
                query = "INSERT INTO user_wallets (user_id, username, referrer_id) VALUES (?, ?, ?)"
                cursor.execute(query, (user_id, username, referrer_id))
            else:
                query = "INSERT INTO user_wallets (user_id, username) VALUES (?, ?)"
                cursor.execute(query, (user_id, username))
            conn.commit()

    
        
            
            

    def count_referrals(self, user_id):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            count = cursor.execute("SELECT COUNT(id) FROM user_wallets WHERE referrer_id =?", (user_id,)).fetchone()[0]
            return count

    def update_paid_subscription(self, user_id, tokens):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE user_wallets SET paid_subscription = 1, purchased_tokens = ? WHERE user_id = ?", (tokens, user_id))
            conn.commit()
            
            
    def cancel_subscription(self, user_id):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE user_wallets SET payment_method_id = NULL WHERE user_id = ?",
                (user_id,)
            )
            conn.commit()

    def count_paid_referrals(self, user_id):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            count = cursor.execute(
                "SELECT COUNT(id) FROM user_wallets WHERE referrer_id = ? AND payment_method_id IS NOT NULL",
                (user_id,)
            ).fetchone()[0]
            return count
    
            

    def check_paid_subscription(self, user_id):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            result = cursor.execute("SELECT paid_subscription, purchased_tokens FROM user_wallets WHERE user_id = ?", (user_id,)).fetchone()
            return result if result else (False, 0)
            
    def save_payment_method_id(self, user_id, payment_method_id):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE user_wallets SET payment_method_id = ? WHERE user_id = ?", (payment_method_id, user_id))
            conn.commit()

    def get_payment_method_id(self, user_id):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            result = cursor.execute("SELECT payment_method_id FROM user_wallets WHERE user_id = ?", (user_id,)).fetchone()
            return result[0] if result else None
        
            

# Пример использования:
# db = Database('your_database_path.db')
# db.add_user('12345')
# db.update_paid_subscription('12345', 100)
# status, tokens = db.check_paid_subscription('12345')

