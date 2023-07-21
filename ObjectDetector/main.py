import numpy as np
import cv2

# img=cv2.imread('lena.png')
# cv2.imshow('output',img)
# cv2.waitKey(0)
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
#置信度
thres=0.45
#非极大值抑制程度,越小抑制程度越大,本质就是IoU交并比
nms_threshold=0.2
classNames=[]
classFile='coco.names'
#导入类文件
with open(classFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')

print(classNames)
#模型与权重路径
configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath='frozen_inference_graph.pb'
#导入模型和权重
net=cv2.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320, 302)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127,5))
net.setInputSwapRB(True)
while True:
    success, img=cap.read()
    #启动检测并设置置信度标准，同时获取相关信息如id、选框四角像素
    classIds, confs, bbox=net.detect(img,confThreshold=thres)
    # print(classIds,bbox)
    #转为list
    bbox=list(bbox)
    confs=list(np.array(confs).flatten())
    #内置类型转为float
    confs=list(map(float,confs))
    print(type(confs))
    print(type(confs[0]))
    print(confs)
    #设置非极大值抑制，返回你应该的输出的bbox中的索引
    indices=cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold=nms_threshold)
    print(indices)
    for i in indices:
        box=bbox[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,h+y),color=(0,255,0),thickness=2)
        cv2.putText(img,classNames[classIds[i]-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(img,str(round(confs[i]*100,2)),(box[0]+300,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    # if len(classIds)!=0:
    #     #标记选框
    #     for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
    #         cv2.rectangle(img,box,color=(0,255,0),thickness=2)
    #         cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    #         cv2.putText(img,str(round(confidence*100,2)),(box[0]+300,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow('output',img)
    cv2.waitKey(1)
