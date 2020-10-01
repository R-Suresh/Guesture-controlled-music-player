"""
Description : V4 of app. Contains code for GUI app and background CV guesture detection

Authors : Rahul Suresh Babu, Abhimanyu Jain

"""

from tkinter import *  
from PIL import ImageTk,Image  
from pygame import mixer
import cv2
import numpy as np
import time
import random

"""
Description : Class to store templates

"""
class template():

    """
    Description : loads and sets template image and its name

    Params
    -------
    fname : string
        template name
    """
    def __init__(self,fname):
        # use gray scale templates for now
        self.img = cv2.imread(fname+'.jpg',0)
        self.nme = fname
    
    """
    Description : Returns template image

    Returns
    -------
    img : numpy array
        template image
    """
    def image(self):
        return self.img
    
    """
    Description : Returns template name

    Returns
    -------
    nme : string
        template name
    """
    def name(self):
        return self.nme

"""
Description : Class to run app

"""
class app:
    # class to tun the pass

    """
    Description : Init function for class

    """
    def __init__(self):
        pass


    """
    Description : Dummy function for Opencv trackbar

    """
    def nothing(self,x):
        pass


    """
    Description : Performs frame differenciation between two frames

    Params
    ------
    frame : numpy array
        frame captured by camera
    prev frame : numpy array
        prev frame captured by camera
    kernel : numpy array
        kernel for morpological opeartions

    Returns
    -------
    temp_holder : numpy array
        Applies morpholgy and blurring operations on the frame difference
    """
    def perform_frame_diff(self,frame,prev_frame, kernel):
        # can add skin color detection here
        temp_holder = cv2.medianBlur(frame,7) - cv2.medianBlur(prev_frame,7)
        temp_holder = cv2.erode(temp_holder,kernel,iterations = 5)
        return cv2.dilate(temp_holder,kernel,iterations = 2)


    """
    Description : Helpes dynamically set HSV limits, threshold limits for skin detection. Saves limits as class params

    """
    def set_skin_params(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('threshold')
        cv2.createTrackbar('limit','threshold',50,255,self.nothing)

        cv2.namedWindow('skin detection')
        cv2.createTrackbar('Hue_limit_lower','skin detection',0,180,self.nothing)
        cv2.createTrackbar('Saturation_limit_lower','skin detection',50,255,self.nothing)
        cv2.createTrackbar('Value_limit_lower','skin detection',30,255,self.nothing)
        cv2.createTrackbar('Hue_limit_upper','skin detection',33,180,self.nothing)
        cv2.createTrackbar('Saturation_limit_upper','skin detection',255,255,self.nothing)
        cv2.createTrackbar('Value_limit_upper','skin detection',115,255,self.nothing)
        
        while(True):
            ret, frame = cap.read()
            cv2.imshow('raw image',frame)

            LB_0 = cv2.getTrackbarPos('Hue_limit_lower','skin detection')
            LB_1 = cv2.getTrackbarPos('Saturation_limit_lower','skin detection')
            LB_2 = cv2.getTrackbarPos('Value_limit_lower','skin detection')
            UB_0 = cv2.getTrackbarPos('Hue_limit_upper','skin detection')
            UB_1= cv2.getTrackbarPos('Saturation_limit_upper','skin detection')
            UB_2 = cv2.getTrackbarPos('Value_limit_upper','skin detection')
            
            LB = np.array([LB_0,LB_1,LB_2],dtype="uint8")
            UB = np.array([UB_0,UB_1,UB_2], dtype = "uint8")

            T = cv2.getTrackbarPos('limit','threshold')
            skinHSV,thresholded = self.obtain_thresholded_skin(frame,UB,LB,T)
            cv2.imshow('skin detection',skinHSV)
            cv2.imshow('threshold',thresholded)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        self.LB = LB
        self.UB = UB
        self.T = T
        cv2.destroyAllWindows()
    
    """
    Description : Performs skin masking and thresholding

    Params
    ------
    frame : numpy array
        frame captured by camera
    UB : list
        list of upper bound HSV limits
    LB : list
        list of lower bound HSV limits
    T : int
        limit for thresholding

    Returns
    -------
    skinHSV : numpy array
        The skin only image
    thresholded : numpy array
        Thresholded version of skin only image
    """
    def obtain_thresholded_skin(self,frame,UB,LB,T):
        imageHSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        skinMask = cv2.inRange(imageHSV,LB,UB)
        skinHSV = cv2.bitwise_and(frame, frame, mask=skinMask)

        kernel = np.ones((5,5),np.uint8)
        skinHSV = cv2.erode(skinHSV,kernel,iterations = 2)

        kernel = np.ones((11,11),np.uint8)
        skinHSV = cv2.dilate(skinHSV,kernel,iterations = 2)

        kernel = np.ones((11,11),np.uint8)
        skinHSV = cv2.erode(skinHSV,kernel,iterations = 1)
    
        skin_gray = cv2.cvtColor(skinHSV, cv2.COLOR_BGR2GRAY)
        M =255

        method=cv2.THRESH_BINARY
        _,thresholded=cv2.threshold(skin_gray, T,M, method)

        return skinHSV,thresholded

    """
    Description : Performs contour detection on thresholded images

    Params
    ------
    thresholded : numpy array
        thresholded skin image

    Returns
    -------
    cont_sorted : numpy array
        contours sorted in decreasing order of size
    """
    def detect_contours(self,thresholded):

        contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        return cont_sorted

    """
    Description : Sleep function, causes delay in between detections

    Params
    ------
    sec : int
        number of seconds to sleep for   
    disp : boolean
        toogles time display on and off
    """
    def time_sleep(self,sec,disp):
        if(disp):
            print('sleeping')
        time.sleep(sec)
        if(disp):
            print('Done sleeping')

    """
    Description : Runs app the detects guestures and maintains GUI

    Params
    ------
    debug : boolean
        toggles intermediate image display windows   
    gui : boolen
        toggles GUI on and off
    time_disp : boolean
        toogles time display on and off
    """
    def run(self,gui,time_disp=False,debug=False):

        if(gui):
            # starting GUI
            root = Tk()  
            canvas = Canvas(root, width = 900, height = 600)  
            canvas.pack()  
            img = ImageTk.PhotoImage(Image.open("album_images\old_town_road.png"))

            # loading music
            mixer.init()
            mixer.music.load('songs/old_town_road.mp3')  
            mixer.music.play()
            canvas.create_image(20, 20, anchor=NW, image=img) 
            play_flag = 1

        # acquire camera
        cap = cv2.VideoCapture(0)
        
        # initial code for motion energy
        i = 1
        diff = None
        flag = 0  # represents reset for motion energy
        _, prev_frame = cap.read()  # starter frame for motion energy
        kernel = np.ones((5,5),np.uint8)

        # loop that maintains GUI and detects guestures
        while(True):

            # resets motion energy after 90 frames - 1.5 seconds
            if( (i%90) == 0):
                flag = 0
                i = 1
                continue
            i = i + 1

            # capture image
            ret, frame = cap.read()
            
            if(flag == 0):
                # reset motion energy 
                diff = self.perform_frame_diff(frame,prev_frame,kernel)
                flag = 1
                prev_frame = frame
            else:
                # continue motion energy
                diff = diff + self.perform_frame_diff(frame,prev_frame,kernel)
                prev_frame = frame

            

            ### Swiping (motion energy detection) starts here
            
            temp_diff = diff.copy()
            lower,upper =200,255  # lower and upper bounds for thresholding

            temp_diff = cv2.cvtColor(temp_diff, cv2.COLOR_BGR2GRAY)
            method=cv2.THRESH_BINARY
            _,thresholded_diff = cv2.threshold(temp_diff,lower,upper, method)
            cont_sorted_diff = self.detect_contours(thresholded_diff)
            
            if(debug):
                # shows the motion energy picture
                cv2.imshow('diff',thresholded_diff)

            if len(cont_sorted_diff) != 0:
                # if any motion is detected

                _,_,w_diff,h_diff = cv2.boundingRect(cont_sorted_diff[0])
                # print('height diff',h_diff)
                # print('width diff',w_diff)
                # print('Ratio diff',w_diff/h_diff)
                ratio_diff = w_diff/h_diff

                if(w_diff>350 and ratio_diff>1.4):
                    if(gui):
                        mixer.music.set_volume(mixer.music.get_volume() + 0.25)
                    print('swipe sideways')
                    flag = 0 # reset motion energy
                    self.time_sleep(3,time_disp) # to prevent multiple detections
                    continue

            ### Swiping ends here
            
            # get skin only and skin thresholded image
            skinHSV,thresholded = self.obtain_thresholded_skin(frame,self.UB,self.LB,self.T)

            if(debug):
                # show captured frame
                cv2.imshow('raw image',frame)
                # show skin hsv
                cv2.imshow('Skin hsv',skinHSV)
                # show thresholded skin
                cv2.imshow('thresholded skin',thresholded)

            ### template matching starts here

            if(self.template_matching(thresholded)!="None"):
                print("Palm Match found")
                if(gui):
                    mixer.music.set_volume(mixer.music.get_volume() - 0.25)
                flag = 0 # reset motion energy
                self.time_sleep(3,time_disp) # to prevent multiple detections
                continue

            ### template matching ends here

            ### pointing direction detection
            cont_sorted = self.detect_contours(thresholded)     # get contours of detections       
            if len(cont_sorted) == 0:
                continue
            x,y,w,h = cv2.boundingRect(cont_sorted[0])
            ratio = w/h
            # print("width",w)
            # print('Height',h)
            # print('overall ratio',w/h)

            # if number of white pixels in threshold_diff > 75 % continue as its possibly a swiping
            n_white_pix = np.sum(thresholded_diff == 255)
            if( (n_white_pix/(thresholded_diff.shape[0] * thresholded_diff.shape[1])) > 0.75 ):
                continue

            if(h>190 and ratio<1):
                print('pointing up')
                if(gui):
                    if(play_flag == 1):
                        mixer.music.pause()
                        play_flag = 0
                    else:
                        mixer.music.unpause()
                        play_flag = 1
                flag = 0 # reset motion energy
                self.time_sleep(3,time_disp) # to prevent multiple detections
                continue
            elif(w>290 and ratio>1.7 and ratio < 2.4):
                print('pointing sideways')
                if(gui):
                    ran_num = random.randint(0, len(songs)-1)
                    img = ImageTk.PhotoImage(Image.open("album_images/"+songs[ran_num]+".png"))
                    mixer.music.load('songs/'+songs[ran_num]+'.mp3')  
                    mixer.music.play()
                    canvas.create_image(20, 20, anchor=NW, image=img) 
                    play_flag = 1
                    root.update_idletasks()
                    root.update()
                flag = 0 # reset motion energy
                self.time_sleep(3,time_disp) # to prevent multiple detections
                continue

            # break condition for loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # update GUI in background
            if(gui):
                root.update_idletasks()
                root.update()

        cap.release()
        cv2.destroyAllWindows()

    """
    Description : Helper function, executes template Matching given template and images

    Params
    ------
    img : numpy array
        scene image

    temp : numpy array
        template image

    Returns
    -------
    flag : Boolean
        True if match occurs, False otherwise
    """
    def helper_template_matching(self,img,temp):
        
        # make sure that image is >= size of template
        if(not(img.shape[0]>temp.shape[0] and img.shape[1]>temp.shape[1])):
            return False

        # Apply NCC template Matching
        meth = 'cv2.TM_CCORR_NORMED'
        method = eval(meth)
        res = cv2.matchTemplate(img, temp,method)
        
        threshold = 0.8  # confidence threshold
        if np.amax(res) >= threshold:
            return True

        return False

    """
    Description : Performs multiscale template matching given template and images

    Params
    ------
    thresholded : numpy array
        thresholded skin image

    Returns
    -------
    occurance : String
        Template name if match occurs, "None" otherwise
    """
    def template_matching(self,thresholded):
        count  = 0
        for template in templates:
            img = thresholded.copy()
            while(img.shape[0]>template.image().shape[0] and img.shape[1]>template.image().shape[1] and count<6):
                count = count + 1
                img = cv2.resize(img,None,fx=0.5,fy=0.5)
                if(self.helper_template_matching(img,template.image())):
                    return template.name()

        return "None"

# get tmplates
templates = []
templates.append(template('vertical_hand4'))

# get songs
songs = ['baam','timber','old_town_road', '10000_hours', 'Ballin', 'Blinding_Lights', 'Circles',
 'DaBaby', 'Dance_Monkey', 'Dont_start_now', 'Everything_I_Wanted', 'Good_As_Hell', 'Heartless',
  'Life_Is_Good', 'Lose_you_to_love_me', 'Memories', 'Roxanne', 'Someone_you_loved', 'The_Box',
   'timber', 'Yummy']

# run app
App = app()
App.set_skin_params()
App.run(gui=True,debug=False)