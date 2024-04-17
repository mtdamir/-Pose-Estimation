import cv2
import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose(enable_segmentation=True)
mpDraw = mp.solutions.drawing_utils


def position_data(lmlist):
    # TODO: right thumb dar global va pain tar baraye test hast v hazv bshe
    global right_ankle, left_ankle, right_heel, left_heel, right_foot_index, left_foot_index, right_thumb
    # shast dast rast
    right_thumb = (lmlist[22][0], lmlist[22][1])
    # ghoozak pa
    right_ankle = (lmlist[28][0], lmlist[28][1])
    left_ankle = (lmlist[27][0], lmlist[27][1])
    # pashne pa
    right_heel = (lmlist[30][0], lmlist[30][1])
    left_heel = (lmlist[29][0], lmlist[29][1])
    # angosht pa
    right_foot_index = (lmlist[32][0], lmlist[32][1])
    left_foot_index = (lmlist[31][0], lmlist[31][1])


# TODO : age estefade nashod hazf bshe
def calculateDistance(p1, p2):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2)
    return length


def overlay(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()

    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))

    mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y + h, x:x + w]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))

    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    bg_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

    return bg_img


def main():
    # -1 choon tasvir shafaf hast?!
    left_shoe = cv2.imread("/home/hossein/Downloads/images/left_shoe.png", -1)
    right_shoe = cv2.imread("/home/hossein/Downloads/images/right_shoe.png", -1)
    # Open the default camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 1000)
    cap.set(4, 720)
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the camera.")
        return

    # Loop to continuously read frames from the camera
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        img = cv2.flip(frame, 1)
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgrgb)
        lmlist = []

        if result.pose_landmarks:

            # draw landmarks on frames
            for id, lm in enumerate(result.pose_landmarks.landmark):
                h, w, c = img.shape
                coorx, coory = int(lm.x * w), int(lm.y * h)
                lmlist.append([coorx, coory])
                # cv2.circle(img,(coorx,coory), 6 ,(0, 255, 0), -1)
                mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)

            # find feet
            # TODO : shayad jaye left v right baraks bashe
            left_foot_ids = [27, 29, 31]
            right_foot_ids = [28, 32, 30]
            left_foot_found = all(result.pose_landmarks.landmark[id].visibility > 0.5 for id in left_foot_ids)
            right_foot_found = all(result.pose_landmarks.landmark[id].visibility > 0.5 for id in right_foot_ids)

            # TODO : shayad in comment eshtba bashe
            # after this func call, we can call foot point by name
            position_data(lmlist)

            # TODO:amir
            # if the model see the right foot show the right shoe
            if right_foot_found:
                right_sole = calculateDistance(right_foot_index, right_heel)
                # x-ghooszak
                centerX = right_ankle[0]
                # y-ghooszak
                centerY = right_ankle[1]
                shoe_size = 2.0
                diameter = round(right_sole * shoe_size)

                x1 = round(centerX - (diameter / 2))
                y1 = round(centerY - (diameter / 2))
                print(
                    f'right_foot_index-x: {right_foot_index[0]}, right_foot_index-y: {right_foot_index[1]},right_heel-x: {right_heel[0]}, right_heel-y: {right_heel[1]} ')
                print(
                    f'right sole : {right_sole}, centerX: {centerX}, centerY: {centerY}, diameter: {diameter}, x1: {x1}, y1: {y1}')
                h, w, c = img.shape

                if x1 < 0:
                    x1 = 0
                elif x1 > w:
                    x1 = w

                if y1 < 0:
                    y1 = 0
                elif y1 > h:
                    y1 = h

                if x1 + diameter > w:
                    diameter = w - x1

                if y1 + diameter > h:
                    diameter = h - y1
                shoe_size = diameter, diameter
                print(f'after x{x1}, y{y1}, diameter{diameter}')
                if (diameter != 0):
                    img = overlay(img, right_shoe, x1, y1, shoe_size)

            # TODO:amir
            if left_foot_found:
                # size kaf pa chap
                left_sole = calculateDistance(left_foot_index, left_heel)
                # x-ghooszak
                centerX = left_ankle[0]
                # y-ghoozak
                centerY = left_ankle[1]

                shoe_size = 3.0
                diameter = round(left_sole * shoe_size)

                x1 = round(centerX - (diameter / 2))
                y1 = round(centerY - (diameter / 2))
                h, w, c = img.shape

                if x1 < 0:
                    x1 = 0
                elif x1 > w:
                    x1 = w

                if y1 < 0:
                    y1 = 0
                elif y1 > h:
                    y1 = h

                if x1 + diameter > w:
                    diameter = w - x1

                if y1 + diameter > h:
                    diameter = h - y1

                shoe_size = diameter, diameter
                if (diameter != 0):
                    img = overlay(img, left_shoe, x1, y1, shoe_size)

            # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        # Display the frame in a window
        cv2.imshow('Webcam', img)

        # Wait for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

