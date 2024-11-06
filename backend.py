import numpy as np
import os
import sounddevice as sd
import sys
from flask import Flask, render_template, jsonify
import threading
import time
from twilio.rest import Client
from datetime import datetime as dt
from datetime import timedelta

outpath = 'data'
if not os.path.exists(outpath):
    os.makedirs(outpath)

# Set parameters
duration = 5  # seconds to update display
sample_rate = 8000
channels = 1
# n = 300  # seconds of audio to display
lookback = 60
min_call_interval = 60 * 5

global last_reset
last_reset = dt.now()

global threshold
threshold = 0.06

# Set up mean volume windows
window_count = int(lookback / duration)
window_size_i = int(duration * sample_rate)
global mean_volumes
mean_volumes = np.zeros([window_count], dtype='float32')
print(mean_volumes.shape)

global last_alerted
last_alerted = None

app_dtype = np.float32

# Flask app setup
app = Flask(__name__)
audio_buffer = np.zeros([0], dtype=app_dtype)
mean_volume = 0

# 40 max

@app.route('/ram_consumption')
def ram_consumption():
    size_mb = audio_buffer.nbytes / 1e6
    return f'{size_mb:.2f} MB'

def save_audio_buffer(audio_buffer):
    np.save(f'{outpath}/{dt.now().strftime("%Y-%m-%d_%H-%M-%S")}.npy', audio_buffer)
    print(f"Audio buffer saved at {dt.now().strftime('%Y-%m-%d_%H-%M-%S')}")

# Background recording function
def update_audio_buffer():
    global audio_buffer
    global last_reset
    global mean_volumes
    while True:
        # Record the audio
        recording = sd.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=channels, 
            dtype='float32'
        )

        sd.wait()
        # Add it to the buffer
        audio_buffer = np.concatenate((
            audio_buffer,
            recording.flatten()
        ))

        mean_volumes = get_mean_volumes(audio_buffer)
        # Decide whether to alert
        alert_or_not(mean_volumes)

        # Only save if buffer has data to avoid duplicate saves
        if dt.now() - last_reset > timedelta(minutes=5) and len(audio_buffer) > 0:
            with threading.Lock():
                print('Saving audio buffer...and resetting!')
                save_audio_buffer(audio_buffer)
                audio_buffer = np.zeros([0], dtype=app_dtype)
                last_reset = dt.now()
                mean_volumes = np.zeros([window_count], dtype='float32')

#%%
def get_twilio_details():
    with open('.secrets', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        twilio_details = {}
        for line in lines:
            s = line.split(' =')
            k = s[0]
            v = s[1].replace("'", '').strip()
            twilio_details[k] = v
    return twilio_details['twilio_auth_token'], twilio_details['twilio_account_sid'], twilio_details['twilio_phone_number']
#%%


# Route for the audio data (JSON format)
@app.route('/audio_data')
def get_audio_data():
    frontend_audio_buffer = audio_buffer[-(lookback * sample_rate):]
    return jsonify(frontend_audio_buffer.tolist())

#%%
def call(recipient_phone_number):
    twilio_auth_token, twilio_account_sid, twilio_phone_number = get_twilio_details()

    client = Client(twilio_account_sid, twilio_auth_token)
    client.calls.create(
        url='https://demo.twilio.com/welcome/voice/',
        to=recipient_phone_number,
        from_=twilio_phone_number
    )

#%%
def alert_or_not(mean_volumes):
    global last_alerted
    global threshold

    # if x of the last y windows are above threshold, alert
    x = 3
    y = 12


    if np.sum(mean_volumes[-y:] > threshold) >= x:
        if last_alerted is None:
            print('First alert! Waking Alice!')
            # Make the alert
            last_alerted = time.time()
            call('+18575077597')
        else:
            time_since_last_alert = time.time() - last_alerted
            if time_since_last_alert > min_call_interval:
                print('Alert! Waking Alice!')
                # Has been more than 5mins since last alert
                last_alerted = time.time()
                # call('+17046041814')
                call('+18575077597')
            else:
                print('Alert! But not enough time has passed since last alert.')
        

def get_mean_volumes(audio_buffer):
    global mean_volumes
    # audio_buffer is a numpy array representing
    # audio_buffer = np.zeros(int(n * sample_rate))  # buffer for last n seconds
    # window_size = 2  # seconds
    allowed_windows = min(window_count, len(audio_buffer) // window_size_i)
    for i in range(allowed_windows):
        start = i * window_size_i
        end = (i + 1) * window_size_i
        strip = audio_buffer[start:end]
        y = np.abs(strip)
        mean_volumes[mean_volumes.shape[0] - i - 1] = np.mean(y)

    return mean_volumes



@app.route('/monitor')
def monitor():
    frontend_audio_buffer = audio_buffer[-(lookback * sample_rate):]
    mean_volumes = get_mean_volumes(frontend_audio_buffer)
    return jsonify(mean_volumes.tolist())


# Main route for displaying the waveform
@app.route('/')
def index():
    return render_template('index.html')

# Run recording in a background thread
threading.Thread(target=update_audio_buffer, daemon=True).start()

# Also run the server
if __name__ == '__main__':
    app.run(port=1237, host='0.0.0.0', debug=True)