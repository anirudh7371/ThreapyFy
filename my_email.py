import yagmail

def send_email(emaill,report,auth):
    email = yagmail.SMTP(user="anirudh7371@gmail.com", password=auth)
    email.send(to= emaill,
               subject="Your Patient's report for today",
               contents=f"Hi Doctor,  \n Check out your patients status today!! \nDo not reply back to this email. \n\n Your patient is having a {report} mood today.\n\nRegards\nMindDiary",)
    return True
