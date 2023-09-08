import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import os

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detectionCon=0.8)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Actual Frame Size:", actual_width, "x", actual_height)

    class DragImg:
        def __init__(self, path, posOrigin, imgType):
            self.posOrigin = posOrigin
            self.imgType = imgType
            self.path = path

            if self.imgType == 'png':
                self.img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
            else:
                self.img = cv2.imread(self.path)
            self.size = self.img.shape[:2]

    path = "ImagesPNG"
    myList = os.listdir(path)
    print(myList)

    listImg = []
    for x, pathImg in enumerate(myList):
        if 'png' in pathImg:
            imgType = 'png'
        else:
            imgType = 'jpg'
        listImg.append(DragImg(f'{path}/{pathImg}', [50 + x * 100, 50], imgType))

    class DragState:
        def __init__(self):
            self.dragging = False
            self.imgObject = None

    dragState = DragState()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=False)

        if hands:
            lmList = hands[0]['lmList']
            length, info, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
            print(length)
            if length < 60:
                cursor = lmList[8]
                if not dragState.dragging:
                    for imgObject in listImg:
                        h, w = imgObject.size
                        ox, oy = imgObject.posOrigin
                        if ox < cursor[0] < ox + w and oy < cursor[1] < oy + h:
                            dragState.dragging = True
                            dragState.imgObject = imgObject
                            break
                elif dragState.imgObject:
                    dragState.imgObject.posOrigin = cursor[0] - dragState.imgObject.size[1] // 2, cursor[1] - dragState.imgObject.size[0] // 2
            else:
                dragState.dragging = False
                dragState.imgObject = None

        try:
            for imgObject in listImg:
                h, w = imgObject.size
                ox, oy = imgObject.posOrigin
                if imgObject.imgType == "png":
                    img = cvzone.overlayPNG(img, imgObject.img, [ox, oy])
                else:
                    img[oy:oy + h, ox:ox + w] = imgObject.img

        except Exception as e:
            print("Error:", str(e))
            pass

        display_width = 1280
        display_height = 720
        resized_img = cv2.resize(img, (display_width, display_height))

        cv2.imshow("Video", resized_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
