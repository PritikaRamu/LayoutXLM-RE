import cv2
import os
import json
import pdb

zh_val = '/home/pritika/funsd_val'

#224,224

with open('error_ner_funsd.json') as json_file:
    error_dict = json.load(json_file)


color_dict = {'B-HEADER':(255,0,0),'I-HEADER':(255,0,0),'B-QUESTION':(0,0,255),'I-QUESTION':(0,0,255),'B-ANSWER':(0,255,0),'I-ANSWER':(0,255,0),'O':(180,105,255)}
''' HEADER: BLUE
    QUESTION: RED
    ANSWER: GREEN
    OTHER: PINK
'''
pdb.set_trace()
file_list = os.listdir(zh_val)
file_list.sort()
count = 0
for file in file_list:
    img1 = cv2.imread(zh_val+'/'+file)
    img2 = cv2.imread(zh_val+'/'+file)
    for i in range(len(error_dict[str(count)]['bbox'])):
        (x1,y1) = (int(error_dict[str(count)]['bbox'][i][0]*img1.shape[1]/1000),int(error_dict[str(count)]['bbox'][i][3]*img1.shape[0]/1000))
        (x2,y2) = (int(error_dict[str(count)]['bbox'][i][2]*img1.shape[1]/1000),int(error_dict[str(count)]['bbox'][i][3]*img1.shape[0]/1000))
        print(error_dict[str(count)]['token'][i])
        img1 = cv2.line(img1,(x1,y1),(x2,y2),color_dict[error_dict[str(count)]['pred'][i]],5)
        img2 = cv2.line(img2,(x1,y1),(x2,y2),color_dict[error_dict[str(count)]['actual'][i]],5)
    count += 1
    cv2.imwrite('funsd_ner/pred/pred_'+file+'.jpg', img1) 
    cv2.imwrite('funsd_ner/true/true_'+file+'.jpg', img2)      


