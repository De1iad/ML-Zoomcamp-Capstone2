from flask import Flask
from flask import request
from flask import jsonify
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor



interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = ['alpine_sea_holly', 'colts_foot', 'lenten_rose', 'rose',
'anthurium', 'columbine', 'lotus', 'ruby-lipped_cattleya',
'artichoke', 'common_dandelion', 'love-in-a-mist', 'siam_tulip',
'azalea', 'corn_poppy', 'magnolia', 'silverbush',
'ball_moss', 'cyclamen', 'mallow', 'snapdragon',
'balloon_flower', 'daffodil', 'marigold', 'spear_thistle',
'barbeton_daisy', 'desert_rose', 'mexican_aster', 'spring_crocus',
'bearded_iris', 'english_marigold', 'mexican_petunia', 'stemless_gentian',
'bee_balm', 'fire_lily', 'monkshood', 'sunflower',
'bird_of_paradise', 'foxglove', 'moon_orchid', 'sweet_pea',
'bishop_of_llandaff', 'frangipani', 'morning_glory', 'sweet_william',
'black-eyed_susan', 'fritillary', 'orange_dahlia', 'sword_lily',
'blackberry_lily', 'garden_phlox', 'osteospermum', 'thorn_apple',
'blanket_flower', 'gaura', 'oxeye_daisy', 'tiger_lily',
'bolero_deep_blue', 'gazania', 'passion_flower', 'toad_lily',
'bougainvillea', 'geranium', 'pelargonium', 'tree_mallow',
'bromelia', 'giant_white_arum_lily', 'peruvian_lily', 'tree_poppy',
'buttercup', 'globe-flower', 'petunia', 'trumpet_creeper',
'californian_poppy', 'globe_thistle', 'pincushion_flower', 'wallflower',
'camellia', 'grape_hyacinth', 'pink-yellow_dahlia', 'water_lily',
'canna_lily', 'great_masterwort', 'pink_primrose', 'watercress',
'canterbury_bells', 'hard-leaved_pocket_orchid', 'poinsettia', 'wild_pansy',
'cape_flower', 'hibiscus', 'primula', 'windflower',
'carnation', 'hippeastrum', 'prince_of_wales_feathers', 'yellow_iris',
'cautleya_spicata', 'japanese_anemone', 'purple_coneflower', 'clematis', 'king_protea', 'red_ginger']


preprocessor = create_preprocessor('xception', target_size=(299,299))

app = Flask('classify_flower')


@app.route('/predict', methods=['POST'])
def predict():
    image_url = request.get_json()
    X = preprocessor.from_url(image_url['url'])
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()
    return jsonify(dict(zip(classes, float_predictions)))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4242)