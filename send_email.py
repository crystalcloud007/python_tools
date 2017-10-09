import smtplib
from email.mime.text import MIMEText

mail_host = ''
mail_user = ''
mail_pass = ''

def sendMail(content,addr,subject):
    msg = MIMEText(content.encode('utf8'),_subtype='plain',_charset='utf8')
    msg['From'] = mail_user
    msg['Subject'] = subject
    msg['To'] = addr

    try:
        print('sending mail....')
        s = smtplib.SMTP_SSL(mail_host,465)
        s.login(mail_user,mail_pass)
        s.sendmail(mail_user,addr,msg.as_string())
        s.close()
        print('completed')

    except Exception as e:
        print('EXCEPTION:' + e)

mail_host = input('please input the mail host. Ordinarilly, it looks like smtp.sitename.com\n>')
mail_user = input('please input your email address\n>')
mail_pass = input('please input the password of your email account\n>')
addr = input('Please input the mail address you want to send to\n>')
subject = input('please input the subject\n>')
content = input('please input the content\n>')
sendMail(content,addr,subject)