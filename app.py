import tensorflow as tf 
from flask import Flask, request, jsonify 
import os  
import numpy as np
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model("/Users/tashwinsj/Desktop/ml deploy/indianFood.h5") 

class_names = ['adhirasam', 'aloo_gobi','aloo_matar','aloo_methi','aloo_shimla_mirch','aloo_tikki','anarsa','ariselu','bandar_laddu','basundi','bhatura','bhindi_masala','biryani','boondi','butter_chicken','chak_hao_kheer','cham_cham','chana_masala','chapati','chhena_kheeri','chicken_razala','chicken_tikka',
 'chicken_tikka_masala','chikki','daal_baati_churma','daal_puri','dal_makhani','dal_tadka','dharwad_pedha','doodhpak','double_ka_meetha','dum_aloo','gajar_ka_halwa','gavvalu','ghevar','gulab_jamun','imarti','jalebi','kachori','kadai_paneer','kadhi_pakoda','kajjikaya','kakinada_khaja','kalakand','karela_bharta',
 'kofta','kuzhi_paniyaram','lassi','ledikeni', 'litti_chokha','lyangcha','maach_jhol','makki_di_roti_sarson_da_saag','malapua','misi_roti','misti_doi','modak','mysore_pak','naan','navrattan_korma','palak_paneer','paneer_butter_masala','phirni','pithe','poha','poornalu', 'pootharekulu','qubani_ka_meetha','rabri',
 'ras_malai','rasgulla','sandesh','shankarpali','sheer_korma','sheera','shrikhand','sohan_halwa','sohan_papdi','sutar_feni','unni_appam']




@app.route('/', methods=['POST'])
def post_example():
      # Assumes the data is in JSON format
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image'] 
    img = Image.open(file)

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        img_arr = np.array(img) 
        imge = tf.convert_to_tensor(img_arr, dtype=tf.float32)
        imge = tf.image.resize(imge , [256,256]) 
        prediction = model.predict(tf.expand_dims(imge, axis =0)) 
        ans = class_names[prediction.argmax()] 
        print(ans) 

        return jsonify({"dish" : ans}), 200 
    else:
        return jsonify({'error': 'Invalid data format'}), 400
    

if __name__ == '__main__':
    app.run(port =8000)