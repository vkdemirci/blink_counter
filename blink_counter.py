import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture(0)

detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)

while True:

    success, cam = cap.read()
    cam, faces = detector.findFaceMesh(cam, draw=False)
    # yüzü tanımlamak için kullanıldı draw = True olursa kod çalıştırılınca görünür hale gelecektir.

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(cam, face[id], 5, color, cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        # göz kapakçığının en üst noktası ile alt noktası arasındaki dikey uzunluk
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)
        # göz kağakçığının en sağ ve sol noktaların aralarındaki yatay uzunluk

        cv2.line(cam, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(cam, leftLeft, leftRight, (0, 200, 0), 3)
        # uzunlukları gösteren çizgiler

        # kameraya yakınlaşıp uzaklaşınca sorun olmaması için açı hesaplaması:
        ratio = int((lenghtVer / lenghtHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 32 and counter == 0:
            blinkCounter += 1
            color = (0, 200, 0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (255, 0, 255)

        cvzone.putTextRect(cam, f'Kirpma Sayaci: {blinkCounter}', (50, 100),
                           colorR=color)
        # kod çalıştırılınca görüldüğü gibi açı değeri belli seviyenin altına düşünce sayacımıza bir ekliyoruz.

        camPlot = plotY.update(ratioAvg, color)
        cam = cv2.resize(cam, (640, 360))  # ekran boyutu sınırlandırıldı.
        camStack = cvzone.stackImages([cam, camPlot], 2, 1)
    else:
        cam = cv2.resize(cam, (640, 360))
        camStack = cvzone.stackImages([cam, cam], 2, 1)

    cv2.imshow("Webcam", camStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # uygulamayı kapatmak için "q" kullanılabilir.
        break
cv2.destroyAllWindows()
