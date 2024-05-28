from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# Эндпоинт для манифеста
@app.route('/tonconnect-manifest.json', methods=['GET'])
def manifest():
    manifest_data = {
        "url": "https://4290-91-107-125-14.ngrok-free.app",  # URL, предоставленный Ngrok
        "name": "16 ngrok app",
        "iconUrl": "https://4290-91-107-125-14.ngrok-free.app/static/artwork1.png",
    }
    return jsonify(manifest_data)

# Эндпоинт для подтверждения транзакций
@app.route('/process_payment', methods=['POST'])
def process_payment():
    payment_data = request.json
    # Обработка данных платежа
    return jsonify({"status": "success"})

# Эндпоинт для отображения главной страницы
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ton_value = request.form.get('ton_value')
        # Здесь можно добавить обработку введенного значения
        print(f"Введенное значение: {ton_value}")
    return render_template('index.html')

# Эндпоинт для отображения страницы "Особенности"
@app.route('/features')
def features():
    return render_template('features.html')

# Эндпоинт для отображения страницы "Цены"
@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

if __name__ == '__main__':
    app.run(debug=True)
