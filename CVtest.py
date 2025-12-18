import cv2
import os

capture = cv2.VideoCapture(1)

photos_dir = r"D:\Wiezmann\First-Proto\Photos"
os.makedirs(photos_dir, exist_ok=True)

f = 0

while True:
    ret, frame = capture.read()
    if not ret:
        break

    cv2.imshow("rahh", frame)

    cv2.imwrite(
        os.path.join(photos_dir, f"Photo{f}.png"),
        frame
    )
    f += 1

    if cv2.waitKey(1)  == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
