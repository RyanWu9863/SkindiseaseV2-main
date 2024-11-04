from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # 允許跨源請求

# 定義症狀標籤
classes = {
    4: ('黑色素細胞痣'),
    6: ('黑色素瘤'),
    2: ('良性角化樣病變'),
    1: ('基底細胞癌'),
    5: ('化膿性肉芽腫和出血'),
    0: ('日光性角化症和上皮內癌'),
    3: ('皮膚纖維瘤')
}

# 加載訓練好的 Keras 模型
model = load_model(r'E:\SkindiseaseV2-main\best_model.keras')  # 替換為您的模型路徑

# 檢查模型結構
model.summary()  # 查看模型的摘要，檢查輸入形狀

def preprocess_image(image):
    image = image.resize((28, 28))  # 調整圖像大小
    image_array = np.array(image)  # 轉為 NumPy 陣列
    image_array = image_array / 255.0  # 像素值縮放
    image_array = np.expand_dims(image_array, axis=0)  # 添加批次維度
    return image_array

@app.route('/upload', methods=['POST'])
def upload_image():
    app.logger.info(f"Received files: {request.files}")
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400

    file = request.files['image']
    app.logger.info(f"Received file: {file.filename}")

    if file.filename == '':
        app.logger.error('未选择任何文件')
        return jsonify({'error': '未选择任何文件'}), 400

    try:
        image = Image.open(file.stream)
    except Exception as e:
        app.logger.error(f'无法处理图片: {str(e)}')
        return jsonify({'error': f'无法处理图片: {str(e)}'}), 400
    
    processed_image = preprocess_image(image)
    
    try:
        predictions = model.predict(processed_image)
        prediction_index = np.argmax(predictions)
        predicted_symptom = classes[prediction_index]  # 獲取症狀名稱
        
        return jsonify({'prediction': predicted_symptom})
    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 400
    
@app.route('/predict_shape', methods=['GET'])
def get_predict_shape():
    dummy_image = np.zeros((1, 28, 28, 3))  # 创建一个假的图像数据
    dummy_prediction = model.predict(dummy_image)
    prediction_shape = dummy_prediction[0].shape
    return jsonify({'shape': prediction_shape})

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
