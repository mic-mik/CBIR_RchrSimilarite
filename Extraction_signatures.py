import cv2
import numpy as np
import face_recognition
import os

# Images path
path = './Images'
# Global variables
images = [] # List of images
classNames = [] # List of image names
# Grab all images from the folder
myList = os.listdir(path)
#print(myList)
# Load images
for img in myList:
    curImg = cv2.imread(os.path.join(path, img))
    images.append(curImg)
    imgName = os.path.splitext(img)[0]
    classNames.append(imgName)

# Define find face and encode function
def findEncodings(img_List, imgName_List):
    """_summary_

    Args:
        img_List (_type_): _description_
        imgName_List (_type_): _description_
    """
    signatures = []
    count = 1
    for myImg, name in zip(img_List, imgName_List):
        img = cv2.cvtColor(myImg, cv2.COLOR_BGR2RGB)
        signature = face_recognition.face_encodings(img)[0]
        signature_class = signature.tolist() + [name]
        signatures.append(signature_class)
        print(f'{int((count/(len(img_List)))*100)} % extracted ...')
        count += 1
    face_array = np.array(signatures)
    np.save('FaceSignatutes.npy', face_array)
    print('Signature saved')

def main():
    findEncodings(images, classNames)

if __name__ == '__main__':
    main()
    