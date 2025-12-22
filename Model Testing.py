import os
import cv2
import torch.nn as nn
from PIL import Image
from playsound3 import playsound
import torch, torchvision
from torchvision.models import resnet18 
import torchvision.transforms as transforms
from torchvision import transforms as transforms



# Transforms
VAL_TF = transforms.Compose([
    transforms.Resize(256), # resizes
    transforms.CenterCrop(224), # crops to size
    transforms.ToTensor(), # turns the image into a tensor
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), # normalizes the image based on imageNet stats
])

# moving the model to the Gpu
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Decector functions
def ClassifyDistract(img, model):
    with torch.no_grad():
        img = Image.fromarray(img)
        tf_img = VAL_TF(img).unsqueeze(0).to(DEVICE)  # opening the image and transforming it 
        model.to(DEVICE)
        logits = model(tf_img) # the results        
        pred = int(logits.argmax(dim=1).item() ) # the chosen class   
        if (pred == 0):
            print("Drinking") 
        elif (pred == 1):
            print("Doing hair and makeup")
        elif (pred == 2):
            print ( "Using radio")
        elif (pred == 3):
            print( "Reaching behind")
        elif (pred == 4):
            print( "Driving safely")
        elif (pred == 5 or pred == 6):
            print( "Using phone")
        elif (pred == 7):
            print( "Talking to passenger")
        else:
            print( "Texting") 
        
        return pred

def person_present(img,model,score_thr=0.6):
    with torch.no_grad():
        tf_img = transforms.ToTensor()(img).to(DEVICE) # opening the image and transforming it into a tensor
        out = model([tf_img])[0] # passing it through the model
        keep = (out['scores'] >= score_thr) & (out['labels'] == 1) # how sure is the model that a person is presant
        return bool(keep.sum().item())

#Custom Model:
class CNN_Distract(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Distract, self).__init__()
        self.step = nn.Sequential(
            #first conv layer
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  #112x112


            #Second conv layer
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  #56x56

            #Third conv layer
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  #28x28


            #Fourth conv layer
            nn.Conv2d(128,256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            
            
            
            nn.AdaptiveAvgPool2d(1),  #turns the outputted tensor into a 256,1,1
            nn.Flatten(), #turns the tensor into a [256]
            nn.Dropout(0.2), #remove 20 precent of neurons
            nn.Linear(256, num_classes) #final linear layer
        )

    def forward(self, x):
        x = self.step(x)
        return x
        
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Person Presant model
    # Loading the correct model with pretrained rates and setting it to eval mode
    person_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(DEVICE).eval()

    # Distract classification Model
    Distract_model_custom = CNN_Distract(10)
    Distract_model_custom.load_state_dict(torch.load(os.path.join(BASE_DIR, "DistractModel1.0.pth")))
    Distract_model_custom.eval()

    # Adding a capture webcam:
    capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture(r"D:\Wiezmann\First-Proto\Vids to test\vid 1- Mixed.mp4")
    # MAIN LOOP
    last_prediction = 4 #initialize to driving safely
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        elif (person_present(frame,person_model)):
            prediction = ClassifyDistract(frame, Distract_model_custom)
            if (prediction != 4 and last_prediction != 4):
                #print("ALERT YOU ARE NOT DRIVING SAFELY")
                playsound(os.path.join(BASE_DIR, "Sounds","beep.mp3"))  
            else:
                #print("Good job")
                x = 1
            last_prediction = prediction
        else:
            #print("NO DRIVER")
            last_prediction = 4

        cv2.imshow("test", frame)

        if cv2.waitKey(1500)  == ord('e'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

