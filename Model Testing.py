import os
import cv2
import torch.nn as nn
from PIL import Image
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
def ClassifyDistractT(img, model):
    with torch.no_grad():
        img = Image.fromarray(img)
        tf_img = VAL_TF(img).unsqueeze(0).to(DEVICE)  # opening the image and transforming it 
        model.to(DEVICE)
        logits = model(tf_img) # the results        
        pred = int(logits.argmax(dim=1).item() ) # the chosen class   
        return pred

def person_present(img,model,score_thr=0.6):
    with torch.no_grad():
        tf_img = transforms.ToTensor()(img).to(DEVICE) # opening the image and transforming it into a tensor
        out = model([tf_img])[0] # passing it through the model
        keep = (out['scores'] >= score_thr) & (out['labels'] == 1) # how sure is the model that a person is presant
        return bool(keep.sum().item())

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))



    # Person Presant model
    # Loading the correct model with pretrained rates and setting it to eval mode
    person_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(DEVICE).eval()

    # Distract classification Model
    Distract_Model_Transfer = resnet18() 
    feats = Distract_Model_Transfer.fc.in_features # number of final features
    Distract_Model_Transfer.fc = nn.Linear(feats, 10) # adding last layer
    Distract_Model_Transfer.load_state_dict(torch.load(os.path.join(BASE_DIR, "DistractModelTransfer.pth")))
    Distract_Model_Transfer.eval()


    # Adding a capture webcam:
    capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture(r"D:\Wiezmann\First-Proto\Vids to test\vid 3- ai.mp4")
    # MAIN LOOP
    last_prediction = 4 #initialize to driving safely
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        elif (person_present(frame,person_model)):
            prediction = ClassifyDistractT(frame, Distract_Model_Transfer)
            if (prediction != 4 and last_prediction != 4):
                print("ALERT YOU ARE NOT DRIVING SAFELY")
            else:
                print("Good job")
            last_prediction = prediction
        else:
            print("NO DRIVER")
            last_prediction = 4

        cv2.imshow("test", frame)

        if cv2.waitKey(150)  == ord('e'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

