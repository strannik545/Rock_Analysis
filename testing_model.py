import argparse
from Prediction_tool import Predict
from PIL import Image

parser = argparse.ArgumentParser(description="Тестирование модели для классификации изображений")
parser.add_argument("-f", "--file", type=str, help="Путь к входному изображению", required=True)
args = parser.parse_args()

img = Image.open(args.file)

rock_id_model = Predict(num_classes=16, model_path='./models/best_model.pth', device='cpu')
smooth_shape_model = Predict(num_classes=2, model_path='./models/smooth_shape.pth', device='cpu')
sharp_edges_model = Predict(num_classes=2, model_path='./models/sharp_edges.pth', device='cpu')
porous_model = Predict(num_classes=2, model_path='./models/porous.pth', device='cpu')
layered_model = Predict(num_classes=2, model_path='./models/layered.pth', device='cpu')
solid_model = Predict(num_classes=2, model_path='./models/solid.pth', device='cpu')
type_model = Predict(num_classes=2, model_path='./models/type_model.pth', device='cpu')

dict_rock = {0: 6,
    1: 9,
    2: 10,
    3: 11,
    4: 12,
    5: 16,
    6: 18,
    7: 19,
    8: 25,
    9: 27,
    10: 28,
    11: 30,
    12: 32,
    13: 35,
    14: 43,
    15: 51}

rock_result = rock_id_model.predict_single_patch(img)
rock_class = dict_rock.get(rock_result['predicted_class'], "Unknown")
rock_confidence = rock_result['confidence']
print(f"Rock Type Prediction: {rock_class} (Confidence: {rock_confidence:.2f})")

smooth_result = smooth_shape_model.predict_single_patch(img)
smooth_class = "Smooth" if smooth_result['predicted_class'] == 1 else "Not Smooth"
smooth_confidence = smooth_result['confidence']
print(f"Surface Smoothness: {smooth_class} (Confidence: {smooth_confidence:.2f})")

sharp_result = sharp_edges_model.predict_single_patch(img)
sharp_class = "Sharp Edges" if sharp_result['predicted_class'] == 1 else "No Sharp Edges"
sharp_confidence = sharp_result['confidence']
print(f"Sharp Edges: {sharp_class} (Confidence: {sharp_confidence:.2f})")

porous_result = porous_model.predict_single_patch(img)
porous_class = "Porous" if porous_result['predicted_class'] == 1 else "Not Porous"
porous_confidence = porous_result['confidence']
print(f"Porosity: {porous_class} (Confidence: {porous_confidence:.2f})")

layered_result = layered_model.predict_single_patch(img)
layered_class = "Layered" if layered_result['predicted_class'] == 1 else "Not Layered"
layered_confidence = layered_result['confidence']
print(f"Layering: {layered_class} (Confidence: {layered_confidence:.2f})")

solid_result = solid_model.predict_single_patch(img)
solid_class = "Solid" if solid_result['predicted_class'] == 1 else "Not Solid"
solid_confidence = solid_result['confidence']
print(f"Solidity: {solid_class} (Confidence: {solid_confidence:.2f})")

type_result = type_model.predict_single_patch(img)
type_class = "Urate + calcium" if type_result['predicted_class'] == 1 else "Calcium"
type_confidence = type_result['confidence']
print(f"Chemical composition of the stone: {type_class} (Confidence: {type_confidence:.2f})")


