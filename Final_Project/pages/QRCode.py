import pyqrcode
qr=pyqrcode.create('Hello')
qr.png('hello.png', scale=7)
