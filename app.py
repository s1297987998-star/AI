import base64
import io
import json
import os
from types import SimpleNamespace

from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

from model.SARB import SARB
from utils.checkpoint import load_pretrained_model
from utils.dataloader import get_data_path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-secret'

DATASET = 'COCO2014'
PRETRAINED_BACKBONE = './data/checkpoint/resnet101.pth'
CHECKPOINT_PATH = './exp/checkpoint/SARB-COCO-SupCon/Checkpoint_Best.pth'
WORD_FEATURE_PATH = './data/coco/vectors.npy'
CATEGORY_MAP_PATH = './data/coco/category.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair dryer', 'toothbrush'
]


def build_args():
    return SimpleNamespace(
        dataset=DATASET,
        classNum=80,
        prob=0.5,
        pretrainedModel=PRETRAINED_BACKBONE,
        resumeModel='None',
        evaluate=False,
        mixupEpoch=5,
        contrastiveLossWeight=0.05,
        supConTemp=0.1,
        prototypeNum=10,
        recomputePrototypeInterval=5,
        isAlphaLearnable=True,
        isBetaLearnable=True,
        post='SARB-COCO-SupCon',
        printFreq=1000,
        epochs=20,
        startEpoch=0,
        stepEpoch=10,
        batchSize=1,
        lr=3e-5,
        momentum=0.9,
        weightDecay=5e-4,
        cropSize=448,
        scaleSize=512,
        workers=0,
    )


def get_category_names():
    return COCO_CLASS_NAMES


def compute_graph(labels):
    graph = np.zeros((labels.shape[1], labels.shape[1]), dtype=np.float32)
    for index in range(labels.shape[0]):
        indexs = np.where(labels[index] == 1)[0]
        for i in indexs:
            for j in indexs:
                graph[i, j] += 1
    for i in range(labels.shape[1]):
        if graph[i, i] != 0:
            graph[i] /= graph[i, i]
    np.nan_to_num(graph, copy=False)
    return graph


def build_model():
    args = build_args()
    train_dir, train_anno, train_label, _, _, _ = get_data_path(args.dataset)

    if not os.path.exists(train_label):
        raise FileNotFoundError(f'Train label file not found: {train_label}')

    labels = np.load(train_label)
    graph = compute_graph(labels)
    word_features = np.load(WORD_FEATURE_PATH)
    model = SARB(graph, word_features,
                 prototypeNum=args.prototypeNum,
                 classNum=args.classNum,
                 isAlphaLearnable=args.isAlphaLearnable,
                 isBetaLearnable=args.isBetaLearnable)

    if os.path.exists(args.pretrainedModel):
        model = load_pretrained_model(model, args)

    if os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        except TypeError:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.BICUBIC),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return tensor


def image_to_base64(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def get_sample_image(index: int):
    _, _, _, test_dir, test_anno, _ = get_data_path(DATASET)

    if not os.path.exists(test_anno) or not os.path.isdir(test_dir):
        raise FileNotFoundError('COCO validation annotations or image directory not found. 请下载并解压 COCO 数据集，或直接上传图片进行预测。')

    with open(test_anno, 'r', encoding='utf-8', errors='ignore') as f:
        dataset = json.load(f)

    images = dataset.get('images', [])
    if not images:
        raise ValueError('COCO annotation文件中未找到 images 字段。')

    index = max(0, min(index, len(images) - 1))
    img_info = images[index]
    image_path = os.path.join(test_dir, img_info['file_name'])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'测试图片文件未找到: {image_path}')

    image = Image.open(image_path).convert('RGB')
    image_id = img_info['id']
    categories = [ann['category_id'] for ann in dataset.get('annotations', []) if ann.get('image_id') == image_id]
    category_names = get_category_names()
    label_vector = np.zeros(80, dtype=np.float32)
    # 如果没有 annotation，actual_labels 为空
    for cat in set(categories):
        idx = int(cat) - 1
        if 0 <= idx < len(category_names):
            label_vector[idx] = 1
    actual_labels = [category_names[i] for i in range(len(label_vector)) if label_vector[i] == 1]
    return image, actual_labels


def predict(image: Image.Image, model, category_names):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = torch.sigmoid(output).squeeze(0).cpu().numpy()
    topk = probabilities.argsort()[::-1][:10]
    return [{'name': category_names[i], 'score': float(probabilities[i])} for i in topk]


MODEL = None
CATEGORY_NAMES = None
MODEL_STATUS = ''


def initialize():
    global MODEL, CATEGORY_NAMES, MODEL_STATUS
    CATEGORY_NAMES = get_category_names()
    try:
        MODEL = build_model()
        checkpoint_found = os.path.exists(CHECKPOINT_PATH)
        backbone_found = os.path.exists(PRETRAINED_BACKBONE)
        status_parts = []
        status_parts.append(f'Backend ready on {DEVICE}')
        status_parts.append('Checkpoint loaded' if checkpoint_found else 'Checkpoint not found')
        status_parts.append('Backbone pretrained weights loaded' if backbone_found else 'Backbone weights not found')
        MODEL_STATUS = ' | '.join(status_parts)
    except Exception as ex:
        MODEL_STATUS = f'Error initializing model: {ex}'
        MODEL = None


@app.route('/', methods=['GET', 'POST'])
def index():
    global MODEL, CATEGORY_NAMES, MODEL_STATUS
    if CATEGORY_NAMES is None:
        initialize()
    predictions = []
    image_data = None
    actual_labels = []
    sample_index = request.form.get('sample_index', '0')
    error_message = None

    if request.method == 'POST':
        uploaded = request.files.get('image_file')
        try:
            if uploaded and uploaded.filename:
                image = Image.open(uploaded.stream).convert('RGB')
                actual_labels = []
            else:
                index_value = int(sample_index) if sample_index.isdigit() else 0
                image, actual_labels = get_sample_image(index_value)

            image_data = image_to_base64(image)
            if MODEL is not None:
                predictions = predict(image, MODEL, CATEGORY_NAMES)
            else:
                error_message = '模型未正确加载，无法预测。'
        except Exception as ex:
            error_message = f'预测失败：{ex}'

    return render_template('index.html', predictions=predictions,
                           image_data=image_data,
                           actual_labels=actual_labels,
                           model_status=MODEL_STATUS,
                           sample_index=sample_index,
                           error_message=error_message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
