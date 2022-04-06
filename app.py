from flask import Flask, render_template, request, jsonify
from ML import summarizeText

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template("index.html")

@app.route('/api/', methods=['GET'])
def summerize():
    if request.method == 'GET':
        input_text = request.args.get('text_input2', '')
        sum_text = summarizeText(input_text)
        return jsonify(sum_text)



if __name__ == '__main__':
    app.run('localhost', 5000, debug=True)
