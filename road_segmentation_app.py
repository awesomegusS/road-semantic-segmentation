import numpy as np
import gradio as gr
import cv2
import torch
from utils import Encoder, Decoder, enforce_orient
from torchvision import transforms


# Define segmentation function
def segment_road(image):
    """
    Paramters: 
    ----------
    Image: numpy ndarray of input image

    Returns:
    --------
    Image: numpy ndarray of overlaid segmentation mask and original image
    """
    
    # Load best model
    num_classes = 1
    encoder = Encoder()
    decoder = Decoder(num_classes)

    encoder_state_dict = torch.load('./Models/encoder-AMP-12.pth')
    decoder_state_dict = torch.load('./Models/decoder-AMP-12.pth')

    encoder.load_state_dict(encoder_state_dict)
    encoder.eval() # mode for inference
    decoder.load_state_dict(decoder_state_dict)
    decoder.eval()
    
    # Load image
    image = enforce_orient(image) # keep original orientation
    
    # transform image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    trfms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=(256, 256)),
                transforms.Normalize(mean=mean,
                                     std=std)
    ])

    input_img_tensor = trfms(image)
        # make inpute tensor suitable for model
    input_img_tensor = input_img_tensor.unsqueeze(0) # 3x256x256 -> 1x3x256x256
    
    # Inference
    with torch.no_grad():
        vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = encoder(input_img_tensor)
        prediction = decoder(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out)
        prediction = torch.sigmoid(prediction)
    
    # Merge mask with original image
    
        # resize predicted label (segmentation mask) to match original image
    pred_label = (cv2.resize(np.transpose(prediction[0].detach().numpy(), (1, 2, 0)),
                             (3024, 4032))>0.5).astype('float')
    pred_label[pred_label==1.0] = 127.0
    
        # Make prediction label have 3 channels
    pred_label = cv2.merge([np.zeros_like(pred_label),
                            pred_label,
                            np.zeros_like(pred_label)]) # out shape (4032, 3024, 3)

        # overalay mask(predicted label) on original image
    overlay_image = np.copy(image)
    
        # Define transparency (alpha) value for the overlay
    alpha = 0.9  # Adjust as needed for the desired transparency level
    
        # Apply the segmentation mask on the overlay image
    overlay_image[pred_label != 0] = (overlay_image[pred_label != 0] * (1 - alpha) +
                                      pred_label[pred_label != 0] * alpha).astype(np.uint8)
    
    
    # Return merged image
    return overlay_image

# Create and Launch Gradio interface
input_image = gr.Image(label="Upload Image")
output_segmented_image = gr.Image(label="Segmented Image")
gr.Interface(fn=segment_road, inputs=input_image, outputs=output_segmented_image).launch(share=True)



