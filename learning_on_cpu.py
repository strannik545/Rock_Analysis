import pandas as pd
from Classification_library import Learning_rocks
import argparse


#device = 'cpu'

# Создаем парсер аргументов
parser = argparse.ArgumentParser(description="Выбор устройства и файла данных")
parser.add_argument("-d", "--device", type=str, choices=["cpu", "cuda"], help="Выбор устройства для обучения: cpu или cuda", required=True)
parser.add_argument("-f", "--file", type=str, help="Путь к файлу данных (CSV)", required=True)
parser.add_argument("-b", "--batch_size", type=int, default=256, help="Размер батча для обучения (по умолчанию 256)")
parser.add_argument("-e", "--num_epochs", type=int, default=20, help="Количество эпох для обучения (по умолчанию 20)")
parser.add_argument("-n", "--num_workers", type=int, default=16, help="Количество рабочих потоков (по умолчанию 16)")
parser.add_argument("-p", "--path", type=str, default='models/', help="Путь к директории для сохранения моделей (по умолчанию 'models/')")
# Добавляем аргумент для помощи
parser.add_argument("-h", "--help", action="help", help="Опции обучения и использования")


# Парсим аргументы
args = parser.parse_args()  

# Проверяем устройство
device = args.device
print(f"Выбрано устройство: {device}")  

BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
NUM_WORKERS = args.num_workers






data = pd.read_csv(args.file)

#data = pd.read_csv('rock_patches.csv')

df = [
    [6, 0, 0, 1, 1, 0, 'Кальциевый'],
    [9, 0, 1, 1, 1, 0, 'Кальциевый'],
    [10, 1, 0, 1, 0, 1, 'Кальциевый'],
    [11, 0, 1, 0, 1, 1, 'Уратный + кальций'],
    [12, 0, 1, 1, 1, 0, 'Уратный + кальций'],
    [16, 1, 0, 0, 0, 1, 'Кальциевый'],
    [18, 0, 1, 0, 1, 1, 'Уратный + кальций'],
    [19, 1, 1, 0, 1, 0, 'Уратный + кальций'],
    [25, 0, 1, 0, 1, 1, 'Уратный + кальций'],
    [27, 1, 0, 1, 1, 0, 'Уратный + кальций'],
    [28, 1, 0, 1, 1, 0, 'Уратный + кальций'],
    [30, 0, 0, 1, 1, 1, 'Кальциевый'],
    [32, 1, 0, 1, 1, 0, 'Уратный + кальций'],
    [35, 0, 1, 1, 1, 0, 'Уратный + кальций'],
    [43, 1, 0, 0, 0, 1, 'Кальциевый'],
    [51, 1, 1, 0, 0, 1, 'Кальциевый']
]
columns = [
    "stone_id",          # камень (идентификатор камня)
    "smooth_shape",      # Гладкая форма
    "sharp_edges",       # Острые острые углы → острые края
    "porous",            # Пористый
    "layered",           # Слоистый
    "solid", # Сплошной
    'type' #Тип химического состава
]

classification_df =  pd.DataFrame(df, columns=columns)

smooth_shape_data = data.copy()
smooth_shape_data.shape
smooth_shape_data.head()
stone_smooth_shape = dict(zip(classification_df['stone_id'], classification_df['smooth_shape']))
smooth_shape_data['rock_type'] = smooth_shape_data['rock_type'].map(stone_smooth_shape)
model_smooth_shape = Learning_rocks(df = smooth_shape_data, device = device, model_name = f'{path}mooth_shape.pth', BATCH_SIZE = BATCH_SIZE, num_ep = NUM_EPOCHS, num_workers=NUM_WORKERS)
model_smooth_shape.learn_model()
#model_smooth_shape.output_met()
model_smooth_shape.clear_memory()


sharp_edges_data = data.copy()
sharp_edges = dict(zip(classification_df['stone_id'], classification_df['sharp_edges']))
sharp_edges_data['rock_type'] = sharp_edges_data['rock_type'].map(sharp_edges)
model_sharp_edges = Learning_rocks(df = sharp_edges_data, device = device, model_name = f'{path}sharp_edges.pth', BATCH_SIZE = BATCH_SIZE, num_ep = NUM_EPOCHS, num_workers=NUM_WORKERS)
model_sharp_edges.learn_model()
#model_sharp_edges.output_met()
model_sharp_edges.clear_memory()


porous_data = data.copy()
porous = dict(zip(classification_df['stone_id'], classification_df['porous']))
porous_data['rock_type'] = porous_data['rock_type'].map(porous)
porous_edges = Learning_rocks(df = porous_data, device = device, model_name = f'{path}porous.pth', BATCH_SIZE = BATCH_SIZE, num_ep = NUM_EPOCHS, num_workers=NUM_WORKERS)
porous_edges.learn_model()
porous_edges.output_met()
porous_edges.clear_memory()


layered_data = data.copy()
layered = dict(zip(classification_df['stone_id'], classification_df['layered']))
layered_data['rock_type'] = layered_data['rock_type'].map(layered)
layered_edges = Learning_rocks(df = layered_data, device = device, model_name = f'{path}layered.pth', BATCH_SIZE = BATCH_SIZE, num_ep = NUM_EPOCHS, num_workers=NUM_WORKERS)
layered_edges.learn_model()
#layered_edges.output_met()
layered_edges.clear_memory()


solid_data = data.copy()
solid = dict(zip(classification_df['stone_id'], classification_df['solid']))
solid_data['rock_type'] = solid_data['rock_type'].map(solid)
solid_edges = Learning_rocks(df = solid_data, device = device, model_name = f'{path}solid.pth', BATCH_SIZE = BATCH_SIZE, num_ep = NUM_EPOCHS, num_workers=NUM_WORKERS)
solid_edges.learn_model()
#solid_edges.output_met()
solid_edges.clear_memory()

type_data = data.copy()
type_data.shape
type_data.head()
type_shape = dict(zip(classification_df['stone_id'], classification_df['type']))
type_data['rock_type'] = type_data['rock_type'].map(type_shape)
type_model = Learning_rocks(df = type_data, device = device, model_name = f'{path}type_model.pth', BATCH_SIZE = 256, num_ep = 20, num_workers = NUM_WORKERS)
type_model.learn_model()
#type_shape.idx_output()
type_model.clear_memory()

