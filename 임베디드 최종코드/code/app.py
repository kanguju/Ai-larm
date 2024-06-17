from flask import Flask, request

app = Flask(__name__)

@app.route('/alert', methods=['POST'])
def alert():
    data = request.get_json()
    message = data.get('message', 'No message received')
    print(f"Alert received: {message}")
    return 'Alert received', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
