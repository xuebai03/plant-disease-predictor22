from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# 加载模型
def load_model():
    try:
        with open('high_recall_agriculture_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"模型加载错误: {e}")
        return None

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': '模型未正确加载'})
    
    try:
        # 获取前端数据
        data = request.json
        features = [
            float(data['temperature']),
            float(data['humidity']),
            float(data['rainfall']),
            float(data['soil_ph'])
        ]
        
        # 转换为模型输入格式
        input_data = np.array([features])
        
        # 预测
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability[1]),  # 正类概率
            'features_used': ['temperature', 'humidity', 'rainfall', 'soil_ph']
        })
        
    except Exception as e:
        return jsonify({'error': f'预测错误: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)