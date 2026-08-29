import io
import os
import sys

import torch
from flask import Flask, request, Response
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import CRNN
import dataset
import utils

app = Flask(__name__)

# -------------------- 模型加载--------------------
captcha_chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Digital-Letters.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN.CRNN(imgH=64, nc=3, nclass=len(captcha_chars) + 1, nh=256,
                  noise_config=None)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

converter = utils.strLabelConverter(captcha_chars, ignore_case=False)
transform = dataset.ResizeNormalize((64, None))

print(f'模型加载完成, device={device}')

# -------------------- API 路由 --------------------
@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return Response('missing image file', status=400,
                        content_type='text/plain; charset=utf-8')

    file = request.files['image']
    if file.filename == '':
        return Response('empty filename', status=400,
                        content_type='text/plain; charset=utf-8')

    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception:
        return Response('invalid image format', status=400,
                        content_type='text/plain; charset=utf-8')

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(image)

    preds = preds.permute(1, 0, 2)
    _, preds = preds.max(2)
    preds = preds.contiguous().view(-1)
    preds_size = torch.IntTensor([preds.size(0)])
    result = converter.decode(preds.data, preds_size.data, raw=False)

    return Response(result, content_type='text/plain; charset=utf-8')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)