import cv2

#print("Before URL")
cap = cv2.VideoCapture('rtsp://admin:NGL0829@192.168.1.226:80/h264Preview_01_main')
#print("After URL")


a = 1
b = 2
c = True

if a == 2 or b == 2 and c == True:
    print("yes")

# while True:

#     #print('About to start the Read command')
#     ret, frame = cap.read()
#     #print('About to show frame of Video.')
#     cv2.imshow("Capturing",frame)
#     #print('Running..')

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

cap.release()
cv2.destroyAllWindows()