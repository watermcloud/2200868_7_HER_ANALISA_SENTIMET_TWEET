from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
import pickle, re
import sqlite3
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info = {
        'title': LazyString(lambda:'API Documentation for Sentiment Analysis'),
        'version': LazyString(lambda:'1.0.0'),
        'description': LazyString(lambda: 'Dokumentasi API untuk Analisa Sentiment')
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    'headers': [],
    'specs': [
        {
            'endpoint':'docs',
            'route': '/docs.json',
        }
    ],
    'static_url_path': '/flasgger_static',
    'swagger_ui': True,
    'specs_route': '/docs/'
}
swagger = Swagger(app, template=swagger_template, config=swagger_config)

max_features = 5000
tokenizer = Tokenizer(num_words=max_features, split=' ',lower=True)
sentiment = ['negative', 'neutral', 'positive']

def lowercase(s):
    return s.lower()

def remove_punctuation(s):
    s = re.sub('((www\.[^\s]+)|(https?:\/\/[^\s]+)|(http?:\/\/[^\s]+)|(http[^\s]+))',' ', s) #menghapus semua URL
    s = re.sub(r'(?:\\x[A-Fa-f0-9]{2})+', r'', s) #menghapus \xDD substring
    s = re.sub('[^0-9a-zA-Z]+', ' ', s) #menghilangkan semua karakter yang bukan huruf atau angka dan menggantinya dengan spasi.
    s = re.sub(r'\brt\b',' ', s) #menghapus awalan rt
    s = re.sub('gue','saya', s) # Mengganti kata "gue" dengan kata "saya"
    s = re.sub(r'\d+', '', s) #menghapus semua angka
    s = re.sub(r'\buser\b',' ', s) #menghapus kata 'user'
    s = re.sub(r':', ' ', s) #menggantikan karakter : dengan spasi 
    s = re.sub(' +', ' ', s) #menggantikan satu atau lebih spasi berturut-turut dengan satu spasi 
    s = re.sub('\n',' ',s) #menggantikan karakter newline (\n) dengan spasi 
    s = re.sub(r'pic.twitter.com.[\w]+', ' ', s) #menghapus semua tautan Twitter (pic.twitter.com)
    s = s.strip() #menghilangkan whitespace di awal dan di akhir teks
    s = re.sub(r'‚Ä¶', '', s)
    return s

# Review: taruh fungsi-fungsi berururtan
def alay_to_normal(s):
    for word in alay_dict:
        return ' '.join([alay_dict[word] if word in alay_dict else word for word in s.split(' ')])

def cleansing(sent):
    string = lowercase(sent)
    string = remove_punctuation(string)
    string = alay_to_normal(string)
    return string

# Good
db = sqlite3.connect('database.db', check_same_thread = False)
q_kamusalay = 'SELECT * FROM kamusalay'
t_kamusalay = pd.read_sql_query(q_kamusalay, db)
alay_dict = dict(zip(t_kamusalay['alay'], t_kamusalay['normal']))

# Load file sequences rnn
file_rnn = open('RNN/x_pad_sequences.pickle','rb')
feature_file_from_rnn = pickle.load(file_rnn)
file_rnn.close()

# Load file sequences lstm
file_lstm = open('LSTM/x_pad_sequences.pickle','rb')
feature_file_from_lstm = pickle.load(file_lstm)
file_lstm.close()

model_file_from_rnn = load_model('RNN/model.h5')
model_file_from_lstm = load_model('LSTM/model.h5')

# Endpoint RNN teks
@swag_from('docs/RNN_text.yml',methods=['POST'])
@app.route('/RNN_text',methods=['POST'])
def rnn_text():

    original_text = request.form.get('text')

    text = [cleansing(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    guess = pad_sequences(feature,maxlen=feature_file_from_rnn.shape[1])

    prediction = model_file_from_rnn.predict(guess)
    polarity = np.argmax(prediction[0])
    get_sentiment = sentiment[polarity]

    json_response = {
        'status_code': 200,
        'description': 'Result of Sentiment Analysis using RNN',
        'data': {
            'text': text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# Endpoint rnn file
@swag_from('docs/RNN_file.yml',methods=['POST'])
@app.route('/RNN_file',methods=['POST'])
def rnn_file():
    file = request.files["upload_file"]
    df = (pd.read_csv(file, encoding="latin-1"))
    df = df.rename(columns={df.columns[0]: 'text'})
    df['text_clean'] = df.apply(lambda row : cleansing(row['text']), axis = 1)
    
    result = []

    # Review: disini ambil data uji sampel aja, jangan di looping seluruh data.csv | di test pakai data_test.csv
    for index, row in df.iterrows():
        text = tokenizer.texts_to_sequences([(row['text_clean'])])
        guess = pad_sequences(text, maxlen=feature_file_from_rnn.shape[1])
        prediction = model_file_from_rnn.predict(guess)
        polarity = np.argmax(prediction[0])
        get_sentiment = sentiment[polarity]
        result.append(get_sentiment)

    original = df.text_clean.to_list()

    json_response = {
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using LSTM",
        'data' : {
            'text' : original,
            'sentiment' : result
        },
    }
    response_data = jsonify(json_response)
    return response_data

# Endpoint LSTM teks
@swag_from('docs/LSTM_text.yml',methods=['POST'])
@app.route('/LSTM_text',methods=['POST'])
def lstm_text():

    original_text = request.form.get('text')

    text = [cleansing(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    guess = pad_sequences(feature,maxlen=feature_file_from_lstm.shape[1])

    prediction = model_file_from_lstm.predict(guess)
    polarity = np.argmax(prediction[0])
    get_sentiment = sentiment[polarity]

    json_response = {
        'status_code': 200,
        'description': 'Result of Sentiment Analysis using LSTM',
        'data': {
            'text': text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# Endpoint LSTM file
@swag_from('docs/LSTM_file.yml',methods=['POST'])
@app.route('/LSTM_file',methods=['POST'])
def lstm_file():
    file = request.files["upload_file"]
    df = (pd.read_csv(file, encoding="latin-1"))
    df = df.rename(columns={df.columns[0]: 'text'})
    df['text_clean'] = df.apply(lambda row : cleansing(row['text']), axis = 1)
    
    result = []

    # Review: disini ambil data uji sampel aja, jangan di looping seluruh data.csv | di test pakai data_test.csv
    for index, row in df.iterrows():
        text = tokenizer.texts_to_sequences([(row['text_clean'])])
        guess = pad_sequences(text, maxlen=feature_file_from_lstm.shape[1])
        prediction = model_file_from_lstm.predict(guess)
        polarity = np.argmax(prediction[0])
        get_sentiment = sentiment[polarity]
        result.append(get_sentiment)

    original = df.text_clean.to_list()

    json_response = {
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using LSTM",
        'data' : {
            'text' : original,
            'sentiment' : result
        },
    }
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run()
