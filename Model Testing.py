import torch, torchvision
from PIL import Image
from torchvision import transforms as transforms
import os
import torch.nn as nn
import torchvision.transforms as transforms

#moving the model to the Gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#To load  paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Person Presant model
#Loading the correct model with pretrained rates and setting it to eval mode
person_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()


#Distract classification Model
#Transforms
val_tf = transforms.Compose([
    transforms.Resize(256), #resizes
    transforms.CenterCrop(224), #crops to size
    transforms.ToTensor(),#turns the image into a tensor
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),#normalizes the image based on imageNet stats
])

#Transfer model:
from torchvision.models import resnet18, ResNet18_Weights 
Distract_Model_Transfer = resnet18() 
feats = Distract_Model_Transfer.fc.in_features #number of final features
Distract_Model_Transfer.fc = nn.Linear(feats, 10) # adding last layer
Distract_Model_Transfer.load_state_dict(torch.load(os.path.join(BASE_DIR, "DistractModelTransfer.pth")))
Distract_Model_Transfer.eval()


#Decector functions
#Distraction Transfer 
def ClassifyDistractT(img_path):
    with torch.no_grad():
        img = Image.open(img_path)
        x = val_tf(img).unsqueeze(0).to(device)  #opening the image and transforming it 
        Distract_Model_Transfer.to(device)
        logits = Distract_Model_Transfer(x)  #the results        
        pred = int(logits.argmax(dim=1).item() )      #the chosen class   
        return pred


#PersonPresant
def person_present(img_path,score_thr=0.6):
    with torch.no_grad():
        x = transforms.ToTensor()(Image.open(img_path)).to(device) #opening the image and transforming it into a tensor
        out = person_model([x])[0] #passing it through the model
        keep = (out['scores'] >= score_thr) & (out['labels'] == 1) #how sure is the model that a person is presant
        return bool(keep.sum().item())
