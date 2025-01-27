from flask import Flask, jsonify, request
from stable_baselines3 import DQN
import numpy as np

app = Flask(__name__)
model = DQN.load("models/mahjong_dqn_agent")

@app.route('/play', methods=['POST'])
def play():
    data = request.json
    obs = np.array(data["state"])
    action, _ = model.predict(obs)
    return jsonify({'action': int(action)})

if __name__ == '__main__':
    app.run(debug=True)