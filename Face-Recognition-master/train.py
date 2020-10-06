import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
import PIL.Image,PIL.ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from tkcalendar import Calendar
from pymongo import MongoClient


client =  MongoClient(host=['localhost:27017'])
db = client['mylib']
col = db['attandence']

window = tk.Tk()
#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.title("Face_Recogniser")
cv_img = cv2.cvtColor(cv2.imread("11.jpg"), cv2.COLOR_BGR2RGB) # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
height, width, no_channels = cv_img.shape # Create a canvas that can fit the above image
canvas = tk.Canvas(window, width = width, height = height)
# Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
# Add a PhotoImage to the Canvas
canvas.create_image(0, 0, image=photo, anchor=tk.NW)

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
#answer = messagebox.askquestion(dialog_title, dialog_text)
 
#window.geometry('1280x720')
window.configure(background='black')

#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(text="Face Detector", bg="blue",fg="black"  ,width=40  ,height=3,font=('century gothic', 30)) 
message.place(x=100, y=20)

lbl = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="yellow"  ,bg="blue" ,font=('century gothic', 18, ) ) 
lbl.place(x=100, y=200)

txt = tk.Entry(window,width=20  ,bg="white" ,fg="black",font=('century gothic', 15, ' bold '))
txt.place(x=450, y=215)

lbl2 = tk.Label(window, text="Enter Name",width=20  ,fg="yellow"  ,bg="blue"    ,height=2 ,font=('century gothic', 15, )) 
lbl2.place(x=100, y=300)

txt2 = tk.Entry(window,width=20  ,bg="white"  ,fg="black",font=('century gothic', 15,))
txt2.place(x=450, y=315)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="light green"  ,bg="blue"  ,height=2 ,font=('century gothic', 15, )) 
lbl3.place(x=100, y=400)

message = tk.Label(window, text="" ,bg="white"  ,fg="green"  ,width=30  ,height=2, activebackground = "yellow" ,font=('century gothic', 15, )) 
message.place(x=450, y=400)

lbl3 = tk.Label(window, text="Attendance : ",width=20  ,fg="black"  ,bg="blue"  ,height=2 ,font=('century gothic', 15, )) 
lbl3.place(x=100, y=650)

message2 = tk.Label(window, text="" ,fg="black"   ,bg="white",activebackground = "black",width=30  ,height=3  ,font=('century gothic', 15, )) 
message2.place(x=450, y=650)





def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():
   Id=(txt.get())
   name=(txt2.get())
   c = int(Id)
   x = col.find_one({"id":c})
   if(x):
       if (x["id"] == c):
           res = "ID Already exists"
           message.configure(text= res)
   else:
       if(is_number(Id) and name==str(name)):
           cam = cv2.VideoCapture(0)
           harcascadePath = "haarcascade_frontalface_default.xml"
           detector=cv2.CascadeClassifier(harcascadePath)
           sampleNum=0
           while(True):
               ret, img = cam.read()
               gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
               faces = detector.detectMultiScale(gray, 1.3, 5)
               for (x,y,w,h) in faces:
                   cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                   #incrementing sample number
                   sampleNum=sampleNum+1
                   #saving the captured face in the dataset folder TrainingImage
                   cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                   #display the frame
                   cv2.imshow('frame',img)
               #wait for 100 miliseconds
               if cv2.waitKey(100) & 0xFF == ord('q'):
                   break
               # break if the sample number is morethan 100
               elif sampleNum>60:
                   break
           cam.release()
           cv2.destroyAllWindows()
           name=''.join(i for i in name if not i.isdigit())
           res = "Images Saved for ID : " + Id +" Name : "+ name
           row = [Id , name]
           with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
               writer = csv.writer(csvFile)
               writer.writerow(row)
           csvFile.close()
           message.configure(text= res)
       else:
           if(is_number(Id)):
               res = "Enter name"
               message.configure(text= res)
               c=col.find()
           if(name==str(name)):
               res = "Enter  ID"
               message.configure(text= res)

            
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Name)
    message.configure(text= res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=PIL.Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                c=' '
                
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
      
    
    m=0
    print("aq")
    for x in col.find({"id":Id}):
        print(Id)
        print(x["id"])
        print("going into loop")
      
        if(x["date"]==date):
            
            print(x["date"])
            print(date)
            if(x["presence"]=="1"):
                print("going into second if")
                a=x["intime"]
                print(a)
                a1=a.split(":")
                print(a1)
                for i in range(0 , len(a1)):
                    a1[i]=int(a1[i])
                    print(a1)
                b=datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
                print(b)
                b1=b.split(":")
                print(b1)
                for j in range(0 , len(b1)):
                    b1[j]=int(b1[j])
                if(b1[0]-a1[0])>8:
                    m=1
                    pass
                    
                else:
                    if(x["date"]==date):
                        print("going into else")
                        myquery={"date":date}
                        newvalues={"$set":{"presence":"0"}}
                        col.update_one(myquery,newvalues)
        else:
            pass
   
            
    if(m==0):
        
        col.insert_one({"id": Id, "name": c.join(aa), "intime": timeStamp, "date": date, "presence": "1"})
    
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    message2.configure(text= res)




    
clearButton = tk.Button(window, text="Clear", command=clear  ,fg="yellow"  ,bg="blue"  ,width=10  ,height=2 ,activebackground = "yellow" ,font=('century gothic', 15, ))
clearButton.place(x=800, y=200)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="yellow"  ,bg="blue"  ,width=10  ,height=2, activebackground = "yellow" ,font=('century gothic', 15, ))
clearButton2.place(x=800, y=300)    
takeImg = tk.Button(window, text="Capture", command=TakeImages  ,fg="yellow"  ,bg="blue"  ,width=10  ,height=3, activebackground = "yellow" ,font=('century gothic', 15, ))
takeImg.place(x=100, y=500)
trainImg = tk.Button(window, text="Train Image", command=TrainImages  ,fg="yellow"  ,bg="blue"  ,width=10  ,height=3, activebackground = "yellow" ,font=('century gothic', 15, ))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Track", command=TrackImages  ,fg="yellow"  ,bg="blue"  ,width=10  ,height=3, activebackground = "yellow" ,font=('century gothic', 15, ))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="yellow"  ,bg="blue"  ,width=10  ,height=3, activebackground = "yellow" ,font=('century gothic', 15, ))
quitWindow.place(x=1100, y=500)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'))
copyWrite.tag_configure("superscript", offset=10)

copyWrite.configure(state="disabled",fg="red"  )
copyWrite.pack(side="left")
copyWrite.place(x=800, y=750) 
window.mainloop()
