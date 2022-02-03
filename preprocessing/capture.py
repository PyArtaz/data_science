import time
import cv2
from util import check_folder

# Create your own image database by capturing from a connected camera, e.g., a webcam. The images are saved to their
# respective class folders.

# Specify relative file path of the newly created image database
save_path = '../data_science/dataset/Letters'
# Specify which classes the database should have
Dir = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
       'W', 'X', 'Y', 'Z', 'ENTER', 'SPACE', 'DEL', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
# Specify, how often an image is captured and how many instances are created
seconds = 1
instances = 60

# Preparation: Create video capture and variables for capturing every n seconds
cap = cv2.VideoCapture(0)
pwd = save_path + '/' + Dir[0] + '/'
check_folder(pwd)
print(pwd)
count_diff = 0
dir_count = 0

# FPS setup
fps = cap.get(cv2.CAP_PROP_FPS)
multiplier = fps * seconds
start_timer = time.time()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    show_image = cv2.flip(image, 1)
    cv2.imshow('Capture', show_image)

    # Pause on press of P and break on press of Q
    if cv2.waitKey(5) & 0xFF == ord('p'):
        print("Pause")
        time.sleep(10)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    end_timer = time.time()
    diff = end_timer - start_timer

    if diff > 1.2:
        start_timer = time.time()
        print('Please smile :)')
        print('Character: ' + str(Dir[dir_count]) + '  recording: ' + str(count_diff))

        cv2.imwrite(pwd + 'frame_%d.jpg' % count_diff, image)
        count_diff += 1

    # After the amount of captures specified by instances, change the character
    if count_diff > instances:
        count_diff = 0
        print('Change the character ... You have 5 seconds')
        time.sleep(5)

        # Setup the new folder
        dir_count += 1
        pwd = save_path + '/' + Dir[dir_count] + '/'
        check_folder(pwd)
        print('New pwd: ' + pwd)
        start_timer = time.time()

    # Stop recording after 60 instances of each character in Dir were captured
    if len(Dir) == (dir_count + 1) and count_diff == instances:
        break

cap.release()
