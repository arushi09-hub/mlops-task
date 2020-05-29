#!/usr/bin/env python
# coding: utf-8




import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
host_address="arushishukla09@gmail.com"
host_pass="swift9415581194papa"
guest_address="arushishukla09@gmail.com"
subject="Model Accuracy"
content='''Hi,developer
            You have achieved your desired accuracy for this model
            Accuracy page is attached with this mail'''
message=MIMEMultipart()
message['From']=host_address
message['To']=guest_address
message['Subject']=subject
message.attach(MIMEText(content,'plain'))





#attaching files
from email.mime.base import MIMEBase
from email import encoders
filename="prg.py"
attachment=open('/home/prg.py', 'rb')
p=MIMEBase('application','octet-stream')
p.set_payload((attachment).read())
encoders.encode_base64(p)
p.add_header('Content-Disposition',"attachment; filename= %s" % filename)
message.attach(p)
#attach finished





message.attach(MIMEText('accuracy.txt', 'plain'))
session=smtplib.SMTP("smtp.gmail.com",587)
session.starttls()
session.login(host_address,host_pass)
text=message.as_string()
session.sendmail(host_address,guest_address,text)
session.quit()
print('Successfuly Sent')







