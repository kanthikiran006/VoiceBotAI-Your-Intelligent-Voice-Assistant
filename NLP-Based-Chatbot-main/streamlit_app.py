import streamlit as st
import numpy as np
import json
import random
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from gtts import gTTS
from io import BytesIO
import IPython.display as ipd
from IPython.display import Audio

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('inputs.json').read())
model = keras.models.load_model('chatbot.h5')
lemmatizer = WordNetLemmatizer()


def preprocess_input(user_input):
    user_input_words = nltk.word_tokenize(user_input)
    user_input_words = [lemmatizer.lemmatize(word.lower()) for word in user_input_words]
    input_bag = [1 if word in user_input_words else 0 for word in words]
    return input_bag

# Function to get chatbot response
def get_chatbot_response(user_input):
    input_bag = preprocess_input(user_input)
    prediction = model.predict(np.array([input_bag]))
    predicted_class = classes[np.argmax(prediction)]
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            response = random.choice(intent['responses'])
            return response



# Streamlit app
# Function to convert text to speech using gTTS and save to disk
def text_to_speech(text):
    tts = gTTS(text)
    audio_file = 'audio.wav'  
    tts.save(audio_file)
    return audio_file

from IPython.display import Audio, Javascript
def main():
    st.title('Rubybot App')

    user_input = st.text_input('How can I help?', '')

    # Process user input and generate response
    if st.button('Ask'):
        chatbot_response = get_chatbot_response(user_input)
        st.text(f'Rubybot: {chatbot_response}')
        if chatbot_response:
            speech_audio = text_to_speech(chatbot_response)
            audio_data = open(speech_audio, 'rb').read()
            duration =30
            audio = Audio(data=audio_data, autoplay=True)
            st.write(audio)
            
            # Add a stop button
            stop_button = st.button('Stop')
            if stop_button:
                st.markdown(
                    '''
                    <script>
                        var audioElements = document.getElementsByTagName('audio');
                        for (var i = 0; i < audioElements.length; i++) {
                            audioElements[i].pause();
                        }
                    </script>
                    ''',
                    unsafe_allow_html=True
                )
            
            js_code = f'''
            <script>
                var duration = {duration * 1000};
                var audioElements = document.getElementsByTagName('audio');
                setTimeout(function() {{
                    for (var i = 0; i < audioElements.length; i++) {{
                        audioElements[i].pause();
                    }}
                }}, duration);
            </script>
            '''
            st.markdown(js_code, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
