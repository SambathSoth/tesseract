import requests

url = 'https://app.nanonets.com/api/v2/OCR/Model/78f9ff1e-4b1b-4398-aa98-59d9d0200eb7/LabelFile/'

data = {'file': open('C:\\Users\\Asus\\tesseract\\nanonets\\156472641_500456584282344_4155994539396809812_n.jpg', 'rb')}

response = requests.post(url, auth=requests.auth.HTTPBasicAuth('oLfExyzHRMjVqK_iYTCOJQyPSwP44yLG', ''), files=data)

print(response.text)