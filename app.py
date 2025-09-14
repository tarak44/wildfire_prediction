# app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import traceback

st.set_page_config(page_title="Wildfire Classifier", page_icon="🔥")

# ===============================
# WildfireCNN (must match training)
# ===============================
class WildfireCNN(nn.Module):
    """Custom CNN used during training (same as training script)."""
    def __init__(self, num_classes=2):
        super(WildfireCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # NOTE: training used input size 224 -> after 4 pools: 224/16 = 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# ===============================
# Helpers: load checkpoint robustly
# ===============================
CHECKPOINT_PATH = "wildfire_model.pth"

@st.cache_resource
def load_checkpoint(path: str):
    """
    Load checkpoint in a robust way:
      - Try torch.load with weights_only=False (needed for older full checkpoints).
      - Accept both checkpoints that contain 'model' (full model) and
        those that contain 'model_state_dict' + 'class_names'.
    Returns:
      model (nn.Module), class_names (list)
    """
    try:
        # explicitly set weights_only=False because PyTorch 2.6+ defaults can block pickled objects
        checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        # older torch may not accept weights_only parameter; fall back without it
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
    except Exception as e:
        # show full traceback for debugging in Streamlit
        st.error("Failed to load checkpoint. See details below.")
        st.text(traceback.format_exc())
        raise e

    # If the checkpoint contains a full model object
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], nn.Module):
        model = checkpoint["model"]
        class_names = checkpoint.get("class_names", ["class0", "class1"])
        model.eval()
        return model, class_names

    # If the checkpoint contains state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        class_names = checkpoint.get("class_names", ["class0", "class1"])
        num_classes = len(class_names)
        model = WildfireCNN(num_classes=num_classes)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            # give helpful debugging message
            st.error("Model state_dict keys did not match the WildfireCNN architecture. See details below.")
            st.text(str(e))
            raise
        model.eval()
        return model, class_names

    # Possibly the checkpoint is just a state_dict (not wrapped in a dict)
    if isinstance(checkpoint, dict):
        # assume full state_dict
        try:
            class_names = ["nowildfire", "wildfire"]
            model = WildfireCNN(num_classes=len(class_names))
            model.load_state_dict(checkpoint)
            model.eval()
            return model, class_names
        except Exception:
            pass

    # If we reach here, unsupported format
    raise RuntimeError("Unsupported checkpoint format. Expected dict with 'model' or 'model_state_dict'.")

# Load model + class names and provide user-friendly messages
st.title("🔥 Wildfire Image Classifier")
st.write("Model inference app — loads `wildfire_model.pth` and predicts wildfire vs no-wildfire.")

try:
    model, CLASS_NAMES = load_checkpoint(CHECKPOINT_PATH)
    st.success(f"Loaded model from `{CHECKPOINT_PATH}` (classes: {CLASS_NAMES})")
except Exception as e:
    st.error("Could not load model checkpoint. Check the server logs / trace above.")
    st.stop()

# Ensure model on CPU for Streamlit
device = torch.device("cpu")
model.to(device)

# ===============================
# Transform (matches training preprocessing)
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # same as training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===============================
# Streamlit UI for inference
# ===============================
uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error("Could not open image. Please upload a valid image file.")
        st.text(str(e))
        st.stop()

    st.image(img, caption="Uploaded image", use_column_width=True)

    # Preprocess and run model
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        pred_idx = int(pred.item())
        conf_score = float(conf.item())

    st.markdown("### 🔮 Prediction")
    st.write(f"**Class:** `{CLASS_NAMES[pred_idx]}`")
    st.write(f"**Confidence:** {conf_score:.4f}")

    # Show probability breakdown if binary
    if probs.shape[1] <= 10:
        prob_list = [float(p) for p in probs.squeeze(0)]
        prob_display = {CLASS_NAMES[i]: f"{prob_list[i]:.4f}" for i in range(len(prob_list))}
        st.markdown("**Class probabilities:**")
        st.json(prob_display)

st.markdown("---")
st.markdown("⚠️ This app expects `wildfire_model.pth` in the same folder. The checkpoint should contain either:\n\n"
            "- a full `model` object (saved during training), or\n"
            "- a `model_state_dict` + `class_names` (this is what the training script saves).\n\n"
            "If you still see loading errors, ensure the checkpoint was created by the training script and is not corrupted.")
