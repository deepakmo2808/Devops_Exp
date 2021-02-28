from Fekz import app
import ssl

# context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
# context.load_cert_chain("server.crt", "server.key")
# context.use_privatekey_file('key.pem')
# context.use_certificate_file('cert.pem')

if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0')