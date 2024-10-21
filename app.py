import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
import numpy as np
from mnist_model import SimpleNN  # Import the model from the previous file

# Load the trained model
model = SimpleNN()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Transform for input images
transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Prediction function for Gradio
def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class

# Create the Gradio interface
iface = gr.Interface(fn=predict, inputs=gr.Image(type='pil'), outputs="label", title="MNIST Digit Classifier")

# Launch Gradio app
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
