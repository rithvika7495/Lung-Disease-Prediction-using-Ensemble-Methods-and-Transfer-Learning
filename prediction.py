import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

def get_gradcam_heatmap(model, img_array, layer_name, pred_index=None):
    inceptionv3_model = model.get_layer('inceptionv3')
    
    grad_model = tf.keras.models.Model(
        [inceptionv3_model.input], 
        [inceptionv3_model.get_layer(layer_name).output, inceptionv3_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    conv_outputs = tf.einsum('ijk,k->ijk', conv_outputs, pooled_grads)

    heatmap = tf.reduce_mean(conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
       
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4, save_path=None):
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    if save_path:
        superimposed_img.save(save_path)
    else:
        plt.imshow(superimposed_img)
        plt.axis('off')
        plt.show()

def predict_disease(image_path,ensemble_model):
    image = preprocess_image(image_path)
    prediction = ensemble_model.predict(image)
    predicted_index = np.argmax(prediction) 
    predicted_disease = disease_classes[predicted_index]
    return predicted_disease

disease_classes = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule', 
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

disease_info = {
    'Atelectasis': {
        'Overview': 'Atelectasis is a complete or partial collapse of the entire lung or area (lobe) of the lung.',
        'Symptoms': ['Difficulty breathing', 'Rapid shallow breathing', 'Coughing'],
        'Treatments': ['Chest physiotherapy', 'Bronchoscopy', 'Surgery'],
        'Precautionary Methods': ['Deep breathing exercises', 'Frequent position changes', 'Incentive spirometry']
    },
    'Cardiomegaly': {
        'Overview': 'Cardiomegaly is an enlarged heart, which can be a sign of various heart conditions.',
        'Symptoms': ['Shortness of breath', 'Swelling (edema)', 'Fatigue'],
        'Treatments': ['Medications', 'Surgery', 'Lifestyle changes'],
        'Precautionary Methods': ['Regular exercise', 'Healthy diet', 'Avoiding alcohol and smoking']
    },
    'Consolidation': {
        'Overview': 'Consolidation refers to the filling of lung tissue with liquid instead of air.',
        'Symptoms': ['Cough', 'Fever', 'Shortness of breath'],
        'Treatments': ['Antibiotics', 'Antiviral medications', 'Supportive care'],
        'Precautionary Methods': ['Vaccination', 'Good hygiene', 'Avoiding smoking']
    },
    'Edema': {
        'Overview': 'Edema is swelling caused by excess fluid trapped in the body\'s tissues.',
        'Symptoms': ['Swelling', 'Stretched or shiny skin', 'Increased abdominal size'],
        'Treatments': ['Diuretics', 'Reducing salt intake', 'Compression garments'],
        'Precautionary Methods': ['Elevating the affected area', 'Regular exercise', 'Healthy diet']
    },
    'Effusion': {
        'Overview': 'Effusion is the escape of fluid into a body cavity.',
        'Symptoms': ['Chest pain', 'Cough', 'Difficulty breathing'],
        'Treatments': ['Thoracentesis', 'Pleurodesis', 'Medications'],
        'Precautionary Methods': ['Managing underlying conditions', 'Avoiding infections', 'Regular check-ups']
    },
    'Emphysema': {
        'Overview': 'Emphysema is a lung condition that causes shortness of breath.',
        'Symptoms': ['Shortness of breath', 'Wheezing', 'Chronic cough'],
        'Treatments': ['Bronchodilators', 'Steroids', 'Oxygen therapy'],
        'Precautionary Methods': ['Avoiding smoking', 'Regular exercise', 'Vaccinations']
    },
    'Fibrosis': {
        'Overview': 'Fibrosis is the thickening and scarring of connective tissue, usually as a result of injury.',
        'Symptoms': ['Shortness of breath', 'Chronic dry cough', 'Fatigue'],
        'Treatments': ['Medications', 'Oxygen therapy', 'Lung transplant'],
        'Precautionary Methods': ['Avoiding exposure to pollutants', 'Healthy diet', 'Regular exercise']
    },
    'Infiltration': {
        'Overview': 'Infiltration refers to the diffusion or accumulation of substances not normal to it or in amounts in excess of the normal.',
        'Symptoms': ['Cough', 'Fever', 'Shortness of breath'],
        'Treatments': ['Antibiotics', 'Antiviral medications', 'Supportive care'],
        'Precautionary Methods': ['Vaccination', 'Good hygiene', 'Avoiding smoking']
    },
    'Mass': {
        'Overview': 'A mass in the lung is a growth that can be benign or malignant.',
        'Symptoms': ['Cough', 'Chest pain', 'Weight loss'],
        'Treatments': ['Surgery', 'Chemotherapy', 'Radiation therapy'],
        'Precautionary Methods': ['Regular screenings', 'Avoiding smoking', 'Healthy diet']
    },
    'Nodule': {
        'Overview': 'A lung nodule is a small growth in the lung that is usually benign.',
        'Symptoms': ['Often asymptomatic', 'Cough', 'Shortness of breath'],
        'Treatments': ['Monitoring', 'Surgery', 'Medications'],
        'Precautionary Methods': ['Regular screenings', 'Avoiding smoking', 'Healthy diet']
    },
    'Pleural_Thickening': {
        'Overview': 'Pleural thickening is the thickening of the lining of the lungs.',
        'Symptoms': ['Chest pain', 'Shortness of breath', 'Cough'],
        'Treatments': ['Medications', 'Surgery', 'Pulmonary rehabilitation'],
        'Precautionary Methods': ['Avoiding exposure to asbestos', 'Regular check-ups', 'Healthy lifestyle']
    },
    'Pneumonia': {
        'Overview': 'Pneumonia is an infection that inflames the air sacs in one or both lungs.',
        'Symptoms': ['Cough with phlegm', 'Fever', 'Shortness of breath'],
        'Treatments': ['Antibiotics', 'Antiviral medications', 'Supportive care'],
        'Precautionary Methods': ['Vaccination', 'Good hygiene', 'Avoiding smoking']
    },
    'Pneumothorax': {
        'Overview': 'Pneumothorax is a collapsed lung caused by air leaking into the space between the lung and chest wall.',
        'Symptoms': ['Sudden chest pain', 'Shortness of breath', 'Rapid heart rate'],
        'Treatments': ['Needle aspiration', 'Chest tube insertion', 'Surgery'],
        'Precautionary Methods': ['Avoiding smoking', 'Managing underlying lung conditions', 'Regular check-ups']
    }
}