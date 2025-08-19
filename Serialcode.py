import serial
data = serial.Serial(
                  'COM3',
                   baudrate = 9600,
                   parity=serial.PARITY_NONE,
                   stopbits=serial.STOPBITS_ONE,
                   bytesize=serial.EIGHTBITS,
                   timeout=1 # must use when using data.readline()
                    )
def Send():
        data.write(str.encode('A'))
        print('Sent the Character')
