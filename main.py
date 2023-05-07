
import cv2
import numpy as np
from numpy import cross
import os
path="Card Database (Renamed)\Card Database (Renamed)\ENGLISH\Dynastie card list"
images=[]
Names=[]
myList=os.listdir(path)
orb=cv2.ORB_create(nfeatures=1000)
for cl in myList:
    imgcurrent=cv2.imread(f'{path}/{cl}',0)
    images.append(imgcurrent)
    Names.append(os.path.splitext(cl)[0])
print(Names)
def find(images):
    descriptionlist=[]
    for img in images:
        kp,des=orb.detectAndCompute(img,None)
        descriptionlist.append(des)
    return descriptionlist

def findID(images,deslist,threshold=15):
    kp2,des2=orb.detectAndCompute(images,None)
    bf=cv2.BFMatcher()
    matchList=[]
    finalval=-1
    try:
        for des in deslist:
            matches=bf.knnMatch(des,des2,k=2)
            good=[]
            for m,n in matches:
                if m.distance<0.75*n.distance:
                    good.append(m)
            matchList.append(len(good))
    except:
        pass
    if(len(matchList)!=0):
        if(max(matchList)>threshold):
            finalval=matchList.index(max(matchList))
    return finalval








deslist=find(images)
print(len(deslist))
cap=cv2.VideoCapture(0)
while True:

    suc,img2=cap.read()
    imgOG=img2.copy()
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    id=findID(img2,deslist)
    if(id!=-1):
        cv2.putText(imgOG,Names[id],(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
    cv2.imshow('image1',imgOG)
    cv2.waitKey(1)

