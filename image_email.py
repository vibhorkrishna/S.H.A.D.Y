import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


def SendMail(ImgFileName):
    img_data = open(ImgFileName, 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = 'Fall Alert'
    msg['From'] = 'sample@gmail.com.cc'
    msg['To'] = 'sample@gmail.com.cc'

    text = MIMEText("Fall Detected. Send HELP")
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    msg.attach(image)

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login("sample_sender@gmail.com", "sample_password")
    s.sendmail("sample_sender@gmail.com", "sample_receiver@gmail.com", msg.as_string())
    s.quit()