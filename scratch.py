import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

img_data = open("Version_1_base/GOOGL_2017-10-04.png", 'rb').read()
msg = MIMEMultipart()
msg['Subject'] = 'subject'
msg['From'] = 'e@mail.cc'
msg['To'] = 'e@mail.cc'

text = MIMEText("test")
msg.attach(text)
image = MIMEImage(img_data, name=os.path.basename("Version_1_base/GOOGL_2017-10-04.png"))
msg.attach(image)


s = smtplib.SMTP("smtp.gmail.com", 587)
s.ehlo()
s.starttls()
s.ehlo()
s.login('laxmaxer@gmail.com', 'MaceWindu1')
s.sendmail('Graph Generator', 'als5ev@virginia.edu', msg.as_string())
s.quit()
