from keras.models import load_model
import keras
import os
from keras.preprocessing import image
from Grad_Cam import make_gradcam_heatmap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

model= load_model('model_weights/model_adv.h5')

def Predict(img_path):
    # print("heelo",img_path)
    prediction="helo"
    if(os.path.isfile(img_path)):
        
        img = image.load_img(img_path,target_size=(224,224))
        img = image.img_to_array(img)
        # print("img_dhape",img.shape)
        img = np.expand_dims(img,axis=0)
        # print("ime",img.shape)



        img_array=img/255.0
        heatmap = make_gradcam_heatmap(
            img_array, model, last_conv_layer_name = "dropout_2",
        classifier_layer_names = [
        "flatten",
        "dense",
        "dropout_3",
        "dense_1"
        ])
        
        
        img_org = image.load_img(img_path,target_size=(224,224))
        img_org = image.img_to_array(img_org)

        heatmap = np.uint8(255 * heatmap)

        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # We use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # We create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img_org.shape[1], img_org.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * 0.4 + img_org
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        
        file_name, file_extension = os.path.splitext(img_path)

        
        
        # Save the superimposed image
        save_path = file_name+"_cam.jpg"
        superimposed_img.save(save_path)

        
        
        
#         print(model.predict(img))
#         p=model.predict_classes(img)
        
#         print("____________",p[0][0],"________________")
#         # Display Grad CAM
#         plt.figure(figsize=(8,8))
#         print("original")
#         plt.imshow(img1/255.0)
#         plt.show()
        
        
        
#         img_cam = image.load_img(save_path,target_size=(224,224))
#         img_cam = image.img_to_array(img_cam)
#         print("CAM")
#         plt.figure(figsize=(8,8))
#         plt.imshow(img_cam/255.0)
#         plt.show()
#         #display(Image(save_path))
        



        p=model.predict_classes(img)
        # print("pppppppppp",p[0,0])
        if(p[0,0]==1):
            prediction="Covid Negative"
        elif(p[0,0]==0):
            prediction="Covid Positive"
    return prediction


img_path="NORMAL2-IM-0171-0001.jpeg"
print(Predict(img_path))