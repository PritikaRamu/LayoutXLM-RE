import cv2
import os
import json
import pdb

zh_val = '/home/pritika/zh_val'

#224,224

with open('error_rc.json') as json_file:
    error_dict = json.load(json_file)

color_dict = {'QUESTION':(0,0,255),'ANSWER':(0,255,0),'LINE':(255,0,0)}
''' QUESTION: RED
    ANSWER: GREEN
'''

for i in range(len(error_dict['id'])):
    img1 = cv2.imread(zh_val+'/'+error_dict['id'][i][:-2]+'.jpg')
    img2 = cv2.imread(zh_val+'/'+error_dict['id'][i][:-2]+'.jpg')
    for j in range(len(error_dict['pred'][i])):
        pt1 = [int(error_dict['pred'][i][j][0][0]*img1.shape[1]/1000),int(error_dict['pred'][i][j][0][1]*img1.shape[0]/1000)]
        pt2 = [int(error_dict['pred'][i][j][1][0]*img1.shape[1]/1000),int(error_dict['pred'][i][j][1][1]*img1.shape[0]/1000)]
        pt3 = [int(error_dict['pred'][i][j][2][0]*img1.shape[1]/1000),int(error_dict['pred'][i][j][2][1]*img1.shape[0]/1000)]
        pt4 = [int(error_dict['pred'][i][j][3][0]*img1.shape[1]/1000),int(error_dict['pred'][i][j][3][1]*img1.shape[0]/1000)]
        img1 = cv2.line(img1,(pt1[0],pt1[1]),(pt2[0],pt2[1]),color_dict['QUESTION'],5)
        img1 = cv2.line(img1,(pt3[0],pt3[1]),(pt4[0],pt4[1]),color_dict['ANSWER'],5)
        img1 = cv2.line(img1,(int((pt1[0]+pt2[0])/2),int((pt1[1]+pt2[1])/2)),(int((pt3[0]+pt4[0])/2),int((pt3[1]+pt4[1])/2)),color_dict['LINE'],2)
    cv2.imwrite('error_rc/pred/pred_'+error_dict['id'][i]+'.jpg', img1)
    for j in range(len(error_dict['actual'][i])):
        pt1 = [int(error_dict['actual'][i][j][0][0]*img1.shape[1]/1000),int(error_dict['actual'][i][j][0][1]*img1.shape[0]/1000)]
        pt2 = [int(error_dict['actual'][i][j][1][0]*img1.shape[1]/1000),int(error_dict['actual'][i][j][1][1]*img1.shape[0]/1000)]
        pt3 = [int(error_dict['actual'][i][j][2][0]*img1.shape[1]/1000),int(error_dict['actual'][i][j][2][1]*img1.shape[0]/1000)]
        pt4 = [int(error_dict['actual'][i][j][3][0]*img1.shape[1]/1000),int(error_dict['actual'][i][j][3][1]*img1.shape[0]/1000)]
        img2 = cv2.line(img2,(pt1[0],pt1[1]),(pt2[0],pt2[1]),color_dict['QUESTION'],5)
        img2 = cv2.line(img2,(pt3[0],pt3[1]),(pt4[0],pt4[1]),color_dict['ANSWER'],5)
        img2 = cv2.line(img2,(int((pt1[0]+pt2[0])/2),int((pt1[1]+pt2[1])/2)),(int((pt3[0]+pt4[0])/2),int((pt3[1]+pt4[1])/2)),color_dict['LINE'],2)
    cv2.imwrite('error_rc/actual/actual_'+error_dict['id'][i]+'.jpg', img2)     