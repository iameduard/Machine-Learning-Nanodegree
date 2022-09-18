import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import json


def process_image(img):
    tf_img = tf.convert_to_tensor(img, dtype=tf.float32)
    tf_img = tf.image.resize(tf_img,(224,224))
    tf_img /= 255
    return tf_img.numpy()

def predict(image_path, model, k=5):
    image = Image.open(image_path)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    predict = model.predict(process_image(image))
    prob_classes=list(zip(list(predict[0]),map(str,list(range(102)))))
    prob_classes.sort(reverse=True,key=lambda x:x[0])
    sub=prob_classes[:k]
    return (x[0] for x in sub),(x[1] for x in sub)

def main():
    parser = argparse.ArgumentParser(prog='predict')
    parser.add_argument('image_path', action='store', type=str, help='Image Path')
    parser.add_argument('model',action='store', type=str, help='Model Path')
    parser.add_argument('--top_k',default=5, action='store',type=int, help='Return the top KK most likely classes')
    parser.add_argument('--category_names', default='./label_map.json', action='store',help='Path to a JSON file mapping labels to flower names')
    
    args = parser.parse_args()
    image_path = args.image_path
    model = args.model
    top_k = args.top_k
    category = args.category_names
    print(image_path)
    print(model)
    print(top_k)
    print(category)
    model = tf.keras.models.load_model(model,custom_objects={'KerasLayer':hub.KerasLayer},compile=False)
	
    with open('label_map.json','r') as f:
    	dic_class_names = json.load(f)

    probs, classes = predict(image_path, model, top_k)

    print(f'\nThe Top {top_k} Classes are:\n')
    print('Sorted descending by probability\n')
    for prob, class_name in list(zip(probs,classes)):
        print(f'prob({dic_class_names[str(int(class_name)+1)]})={prob}')

if __name__ == '__main__':

    main()	
