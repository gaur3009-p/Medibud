# Load the model
model.load_state_dict(torch.load('vision_language_model.pth'))
model.eval()

# Inference on a single image-text pair
with torch.no_grad():
    image = ...  # Preprocess image
    text = "Sample biomedical text"
    output = model(image, text)
    prediction = torch.argmax(output, dim=1)
