# details of the lib: https://github.com/ikvk/imap_tools#email-attributes
# install the lib
# pip3 install imap-tools
from imap_tools import MailBox, AND
import xlwt
import pandas as pd

''' By: sandipan ghosh
    email: ghoshm21@gmail.com
    date: 28th Feb 2021
    Get all the email (from, to, date, subject and body from the gmail)
    you need 2 face auth enabled in gmail
'''

imap_url = 'imap.gmail.com'
df = pd.DataFrame()

# login to the gmail using secret key
mailbox = MailBox(imap_url).login('ghoshm21@gmail.com', 'secret key')

# get the number of mail from the set mail folder
# mailbox.folder.set('INBOX')
# print('CURRENT MAIL FOLDER IS: ' + str(mailbox.folder.get()))
# get all  the mails sequence numbers
# mails = mailbox.search('all')
# print(len(mails)) #41727

# get the number of emails and emails from every sub-folders email
for f in mailbox.folder.list():
    try:
        current_folder = f['name']
        print(current_folder)
        mailbox.folder.set(current_folder)
        print('CURRENT MAIL FOLDER IS: ' + str(mailbox.folder.get()))
        mails = mailbox.search('all')
        print(len(mails))
        # read the emails and load into pandas
        try:
            for msg in mailbox.fetch():
                df = df.append([[msg.from_, msg.to, msg.date_str, msg.subject, str(msg.text), str(msg.html)]])
        except Exception as e:
            pass 

    except Exception as e:
        pass 

# set the columns name for the DF
df.columns = ['From', 'To', 'Date', 'Subject', 'msg_text', 'msg_html']
df = df.reset_index(drop=True)
# save the DF to json
df.to_json('/media/sandipan/pi/project_data/all_gmails_output.json',orient='records')
mailbox.logout()
