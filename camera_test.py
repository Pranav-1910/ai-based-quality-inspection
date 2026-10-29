import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera not found.")
    print("Check the USB connection and Windows camera permissions.")
else:
    print("Camera connected! Press Q to close the camera window.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Unable to read camera frame.")
            break

        cv2.imshow("USB Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()