

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
from pymongo import MongoClient
from tkinter import *
import sys
import os,cv2
from tkinter import ttk
from tkinter.ttk import Combobox
from tkcalendar import *
from tkinter import messagebox
import pymongo
from imutils.video import VideoStream



#Connection
client = MongoClient(host=['localhost:27017'])
db=client['mylib']
col=db['attandence']

client1 = MongoClient(host=['localhost:27017'])
db=client['admin']
col1=db['admin']

#Removing Widgets
def remove_widgets_admin():
    lbl1.destroy()
    lbl2.destroy()
    entry1.destroy()
    entry2.destroy()
    button2.destroy()
    button3.destroy()

    

#Training Part
def training():
    

    global message1
    global message2
    global message3
    global lbl3
    global lbl4
    global lbl6
    global lbl7
    global txt1
    global txt2
    

    message1 = Label(root, text="Attendance System"  ,width=20  ,height=2,font=('century gothic', 50)) 
    message1.place(x=100, y=20)

    lbl3 = Label(root, text="Enter ID",width=20  ,height=2  ,fg="white"  ,bg="blue" ,font=('century gothic', 15) ) 
    lbl3.place(x=100, y=200)

    txt1 = Entry(root,width=20  ,bg="white" ,fg="black",font=('century gothic', 15))
    txt1.place(x=400, y=215)

    lbl4 = Label(root, text="Enter Name",width=20  ,fg="white"  ,bg="blue"    ,height=2 ,font=('century gothic', 15)) 
    lbl4.place(x=100, y=300)

    txt2 = Entry(root,width=20  ,bg="white"  ,fg="black",font=('century gothic', 15))
    txt2.place(x=400, y=315)

    lbl6 = Label(root, text="Notification : ",width=20  ,fg="white"  ,bg="blue"  ,height=2 ,font=('century gothic', 15)) 
    lbl6.place(x=100, y=400)

    #lbl9=Label(root,width=20, text="branch", fg="white", bg="blue",height=2, font=('century gothic',15))
    #lbl9.place(x=100,y=375)

    #txt3 = Entry(root,width=20  ,bg="blue" ,fg="white",font=('century gothic', 15))
    #txt3.place(x=400, y=400)
    
    message2 = Label(root, text="" ,bg="yellow"  ,fg="green"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15)) 
    message2.place(x=400, y=400)

    lbl7 =Label(root, text="Attendance : ",width=20  ,fg="white"  ,bg="blue"  ,height=2 ,font=('times', 18, ' bold ')) 
    lbl7.place(x=100, y=640)

    message3 = Label(root, text="" ,fg="green"   ,bg="yellow",activeforeground = "green",width=40  ,height=2  ,font=('times', 18, ' bold ')) 
    message3.place(x=400, y=640)
 
    def clear():
        txt1.delete(0, 'end')    
        res = ""
        message2.configure(text= res)

    def clear2():
        txt2.delete(0, 'end')    
        res = ""
        message2.configure(text= res)    
    
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
        Id=(txt1.get())
        print(Id)
        name=(txt2.get())
        c=int(Id)
        #branch=(txt3.get())
        x=col.find_one({"id":c})
        print(x)
        if(x):
            print(x)
            if(x["id"]==c):
                
                print(x["id"])
                res="ID Already Exists"
                message2.configure(text=res)
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

                with open(r'StudentDetails\StudentDetails.csv','a+') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                csvFile.close()
                message2.configure(text= res)
            else:
                if(is_number(Id)):
                    res = "Enter Name"
                    message2.configure(text= res)
                    c=col.find()
            
                if(name==str(name)):
                    res = "Enter Id"
                    message2.configure(text= res)
    
    def TrainImages():
        recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
        harcascadePath = r"haarcascade_frontalface_default.xml"
        detector =cv2.CascadeClassifier(harcascadePath)
        faces,Id = getImagesAndLabels("TrainingImage")
        recognizer.train(faces, np.array(Id))
        recognizer.save(r"TrainingImageLabel\Trainner.yml")
        res = "Image Trained"#+",".join(str(f) for f in Id)
        message2.configure(text= res)

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
        
        c=' '
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
                    
                    print(aa)
                    print(Id)
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

    
        for x in col.find({"id":Id}):
            print(Id)
            print(x["id"])
            print("going into loop")
        
        
            if(x["date"]==date):
                print("going into first if condition")
                if(x["presence"]=="Present"):
                    print("going into second if")
                    a=x["intime"]
                    print(a)
                    a1=a.split(":")
                    print(a1)
                    for i in range(0 , len(a1)):
                        a1[i]=int(a1[i])
                        print(a1)
                        b=datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
                        b1=b.split(":")
                    for j in range(0 , len(b1)):
                        b1[j]=int(b1[j])
                    if(b1[0]-a1[0])>8:
                        pass
                    else:
                        myquery={"presence":"Present"}
                        newvalues={"$set":{"presence":"Absent"}}
                        col.update_one(myquery,newvalues)
                        m=1
                else:
                    
                    m=1
                
            else:
                
                pass    

        if(m==0):
            col.insert_one({"id": Id, "name": c.join(aa), "intime": timeStamp, "date": date, "presence": "Present"})
    

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

    global manualbutton
    global clearButton
    global clearButton2
    global takeImg
    global trainImg
    global trackImg
    global quitWindow
    global copyWrite
    manualbutton=Button(root,text="Manual Attendance",command=manualattendance,fg="black"  ,bg="orange"  ,width=25  ,height=2 ,activebackground = "Red" ,font=('century gothic', 15, ' bold '))
    manualbutton.place(x=900,y=20)
    clearButton = Button(root, text="Clear", command=clear  ,fg="white"  ,bg="blue"  ,width=10  ,height=2 ,activebackground = "yellow" ,font=('century gothic', 15))
    clearButton.place(x=700, y=200)
    clearButton2 = Button(root, text="Clear", command=clear2  ,fg="white"  ,bg="blue"  ,width=10  ,height=2, activebackground = "yellow" ,font=('century gothic', 15))
    clearButton2.place(x=700, y=300)    
    takeImg = Button(root, text="Capture", command=TakeImages  ,fg="white"  ,bg="blue"  ,width=20  ,height=3, activebackground = "yellow" ,font=('century gothic', 15))
    takeImg.place(x=100, y=500)
    trainImg = Button(root, text="Train", command=TrainImages  ,fg="white"  ,bg="blue"  ,width=20  ,height=3, activebackground = "yellow" ,font=('century gothic', 15))
    trainImg.place(x=300, y=500)
    trackImg = Button(root, text="Track", command=TrackImages  ,fg="white"  ,bg="blue"  ,width=20  ,height=3, activebackground = "yellow" ,font=('century gothic', 15,))
    trackImg.place(x=500, y=500)
    quitWindow = Button(root, text="Quit", command=root.destroy  ,fg="white"  ,bg="red"  ,width=20  ,height=3, activebackground = "yellow" ,font=('century gothic', 15,))
    quitWindow.place(x=900, y=500)
    copyWrite = Text(root, background=root.cget("background"), borderwidth=0,font=('century gothic', 30, 'italic bold underline'))
    copyWrite.tag_configure("superscript", offset=10)

    copyWrite.configure(state="disabled",fg="blue")
    copyWrite.pack(side="left")
    copyWrite.place(x=800, y=750) 
        #root.mainloop()
    

#Complete Training

def remove_widgets_train():
    lbl3.destroy()
    lbl4.destroy()
    lbl6.destroy()
    lbl7.destroy()
    txt1.destroy()
    txt2.destroy()
    message1.destroy()
    message2.destroy()
    message3.destroy()
    manualbutton.destroy()
    clearButton.destroy()
    clearButton2.destroy()
    quitWindow.destroy()
    takeImg.destroy()
    trainImg.destroy()
    trackImg.destroy()
    copyWrite.destroy()
    
#Manual Attandence

def manualattendance():
    #root=tk.Tk()

    remove_widgets_train()

    def date_func(ent):
    
        return ent.get_date()
        

    def clear():
        entry1.delete(0,'end')
        entry2.delete(0,'end')
        combo.delete(0,'end')
    
        res=""

    def manual_update():
        a=int(entry1.get())
        
        b=entry2.get()
        c=str(date_func(ent))
        m=0
        d=combo.get()
        print(a,b,c,d)
        for x in col.find():
            
            print("going into loop")
            
            
            if(x["id"]==a):
            
                if(x["date"]==c):
                    print("going into main condition")
                    z=x["presence"]
                    myquery={"presence":z}
                    newvalues={"$set":{"presence":d}}
                    col.update_one(myquery,newvalues)
                    print(z)
                    m=1
                
                else:
                    print("on that date employee update is not possible")

            else:
                print("no id found")
        if(m==1):
            
            messagebox.showinfo("updated","record sucessfully updated")   
        
     
    def show():
        try:
            df=pd.DataFrame(col.find())
            a=str(ent.get_date())
            mylist = df[df.apply(lambda row: row.astype(str).str.contains(a).any(),axis=1)][['id','name','date','presence']]  
            print(mylist) 
    
        #results['T'] = mylist
            frame=Frame(root,bd=2,relief=SUNKEN)
            results=Scrollbar(frame)
            T=Text(frame,height=50,width=80)
            results.pack(side=RIGHT,fill=Y)
            T.pack(side=RIGHT,fill=Y)
            results.config(command=T.yview)
            T.config(yscrollcommand=results.set)
            T.insert(END,mylist)
            exitb=Button(frame,text="close",command=frame.destroy).pack()
            frame.pack()
        
        
        except:
        #results['T'] = 'There was a problem retrieving that information'       
            T.insert(END,'There was a problem retrieving that information')

    def insert():
        a=int(entry1.get())
    
        b=entry2.get()
        c=str(date_func(ent))
    
        d=combo.get()
        col.insert_one({"id": a, "name": b,  "date": c, "presence": d})
    #results.config(text="record inserted successfully")
        messagebox.showinfo("INSERTION","Record Successfully Inserted")
        
    #root.geometry("650x650")
    #root.title("Manual Attendance")
#root.configure(background="white")


    global label1
    global label2
    global label3
    global entry1
    global entry2
    global button1
    global button2
    global button3
    global button4
    global button5
    global button6
    global ent
    global combo
    global button7
    
    label1=Label(root,text="EmployeeID",font=("arialblack",15,'italic'))
    label1.place(x=20,y=20)
    entry1=Entry(root,font=('arial',12))
    entry1.place(x=170,y=20)
    label2=Label(root,text="EmployeeName",font=("arialblack",15,'italic'))
    label2.place(x=20,y=70)
    entry2=Entry(root,font=('arial',12))
    entry2.place(x=170,y=70)
    label3=Label(root,text="Attendance",font=("arialblack",15,'italic'))
    label3.place(x=20,y=120)

    button1=Button(root,text="update",width=20,height=2,command=manual_update)
    button1.place(x=20,y=350)
    button2=Button(root,text="Exit",command=root.destroy,width=20,height=2)
    button2.place(x=20,y=400)
    button3=Button(root,text="Clear",command=clear,width=20,height=2)
    button3.place(x=20,y=450)
    button4=Button(root,text="Insert",command=insert,width=20,height=2)
    button4.place(x=20,y=500)

    v=['Present','Absent','Leave','Half Day']
    combo=Combobox(root,values=v,font=("arial",11,'italic'))
    combo.place(x=170,y=120)

    button5=Button(root,text="Show Database",command=show,width=15,height=2)
    button5.place(x=20,y=270)

    ent=DateEntry(root,width=15,height=2,bg="blue",fg="red",borderwidth=2,font=("aialblack",11,'bold'))
    ent.place(x=170,y=210)    
    button6=Button(root,text="Click Date",command=date_func(ent),width=15,height=2,font=("arialblack",10,'bold'))
    button6.place(x=20,y=200)

    button7=Button(root,text="Back",width=10  ,fg="black"  ,bg="orange"    ,height=1 ,font=('times', 15, ' bold '),command=remove_widgets_manual)
    button7.place(x=1150,y=10)


    #root.mainloop()

'''lower_frame = tk.Frame(root, bg='white', bd=2)
lower_frame.place(relx=0.7, rely=0.25, relwidth=0.3, relheight=0.5, anchor='n')

bg_color = 'white'
results = tk.Label(lower_frame, anchor='n', justify='left', bd=2)
results.config(font=30, bg=bg_color)
results.place(relwidth=1, relheight=0.6)
'''
    
#Removing Widgets Manual Attendance

def remove_widgets_manual():
    label1.destroy()
    label2.destroy()
    label3.destroy()
    entry1.destroy()
    entry2.destroy()
    button1.destroy()
    button2.destroy()
    button3.destroy()
    button4.destroy()
    button5.destroy()
    button6.destroy()
    ent.destroy()
    combo.destroy()
    button7.destroy()
    training()
    


    

#Completed Manual filling

#Admin

def command1():
    x=col1.find_one()
    print("going to db")
    a=x['username']
    b=x['password']
    if entry1.get()==a and entry2.get()==b:
        remove_widgets_admin()
        training()
        #os.system('python train.py')
        #root.deiconify()
        #top.destroy()
    else:
        messagebox.showinfo("Invalid","Invalid UserName or Password")
        #Label(top,text="Invalid UserName or Password",fg="red", font=("arial",10)).pack()
def command2():
    #top.destroy()
    root.destroy()
    sys.exit()
'''def command3():
    if entry1.get()=='admin' and entry2.get()=='manual':
        manualattendance()
        #os.system('python manualattendance.py')
    #top.destroy()
    else:
        messagebox.showinfo("Invalid","Invalid UserName or Password")
        #Label(top,text="Invalid Username or Password",fg='red',font=('arial',10)).pack()
'''    
root=Tk()
#top= Toplevel()

root.geometry("1280x720")
root.title("LOGIN SCREEN")
#root.configure(background="white")
#photo2 = PhotoImage(file='images.png')
#photo=Label(top,image=photo2,bg="white")
lbl1= Label(root,text="Username:",font=("century gothic",15))
lbl1.place(x=500,y=100)
entry1=Entry(root)
entry1.place(x=630,y=108)
lbl2=Label(root,text="Password:",font=("century gothic",15))
lbl2.place(x=500,y=150)
entry2=Entry(root,show="*")
entry2.place(x=630,y=156)
button2=Button(root,text="LOGIN",bd='5',command=command1,font=("century gothic",10,'bold'))
button2.place(x=560,y=200)
button3=Button(root,text="Exit",bd='5',command=command2,font=("century gothic",10,'bold'))
button3.place(x=660,y=200)
#button4=Button(root,text="Exit",bd='5',font=("arialblack",13,'bold'),command=lambda:command2())
#button4.place(x=200,y=200)
entry2.bind("<Return>",command1)

'''photo.pack()
lbl1.pack()
entry1.pack()
lbl2.pack()
entry2.pack()
button2.pack()
button3.pack()
button4.pack()
'''
#root.title("Main Screen")
#root.configure(background="white")
#root.geometry("900x700")

#root.withdraw()
root.mainloop()

