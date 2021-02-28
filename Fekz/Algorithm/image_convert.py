import base64 
image = open('fake.jpe', 'rb') #open binary file in read mode
image_read = image.read()
image_64_encode = base64.b64encode(image_read)
print(image_64_encode)