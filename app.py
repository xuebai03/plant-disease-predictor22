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
        print("模型加载成功")
        return model
    except Exception as e:
        print(f"模型加载错误: {e}")
        return None

# 在应用启动时加载模型
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': '模型未正确加载'}), 500
    
    try:
        # 获取前端数据
        data = request.get_json()
        if not data:
            return jsonify({'error': '没有接收到数据'}), 400
            
        features = [
            float(data.get('temperature', 0)),
            float(data.get('humidity', 0)),
            float(data.get('rainfall', 0)),
            float(data.get('soil_ph', 0))
        ]
        
        # 转换为模型输入格式
        input_data = np.array([features])
        
        # 预测
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability[1]),  # 正类概率
            'confidence': f"{probability[1]*100:.2f}%",
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'预测错误: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)