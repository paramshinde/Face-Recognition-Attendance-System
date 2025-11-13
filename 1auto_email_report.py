import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import os

# Configuration
SENDER_EMAIL = "rohanpandey1234567891011@gmail.com"
SENDER_PASSWORD = "lplfraofzcjkkauh"  # Use Gmail App Password
RECEIVER_EMAIL = "paramshinde14@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Prepare email
msg = MIMEMultipart()
msg['From'] = SENDER_EMAIL
msg['To'] = RECEIVER_EMAIL
msg['Subject'] = f"Daily Attendance Report - {datetime.now().strftime('%Y-%m-%d')}"

body = "Please find attached today's attendance report."

# Attach Excel file
filename = "attendance.xlsx"
filepath = os.path.abspath(filename)

if not os.path.exists(filepath):
    print(f"[ERROR] File {filepath} does not exist.")
    exit()

attachment = open(filepath, "rb")
part = MIMEBase('application', 'octet-stream')
part.set_payload(attachment.read())
encoders.encode_base64(part)
part.add_header('Content-Disposition', f'attachment; filename={filename}')
msg.attach(part)

# Send email
try:
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.send_message(msg)
    server.quit()
    print("Attendance report sent successfully!")
except Exception as e:
    print(f"[ERROR] Failed to send email: {e}")
