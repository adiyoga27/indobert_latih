from flask import Flask, render_template, request, Response
import subprocess
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'assets/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # code for file upload, similar to previous implementation
    if 'csvFile' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['csvFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv'))
        return 'File uploaded successfully', 200
    else:
        return jsonify({'error': 'Only CSV files are allowed'}), 400
@app.route('/analys')
def analys():
    process = subprocess.Popen(['python', 'indobert_test.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        return error.decode('utf-8'), 500
    else:
        return output.decode('utf-8'), 200
    



@app.route('/run_sentiment')
def run_sentiment():
    process = subprocess.Popen(['python', 'sentiment_indobert.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
    def generate():
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                yield f"data: {output}\n\n"
    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)
