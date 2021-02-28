image_64_encode=b''
import base64
image_64_decode = base64.decodebytes(image_64_encode) 
image_result = open('fake.jpg', 'wb') # create a writable image and write the decoding result
image_result.write(image_64_decode)