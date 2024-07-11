import cv2
import mediapipe as mp
import face_recognition

# 얼굴을 인식할 때 기준이 될 이미지를 로드하고 인코딩
known_face_encodings = []
known_face_names = []

# 예시로 두 명의 인물 이미지를 추가합니다.
# 'person1.jpg', 'person2.jpg'를 이미지 파일 경로로 변경하십시오.
image_person1 = face_recognition.load_image_file('person1.jpg')
face_encoding_person1 = face_recognition.face_encodings(image_person1)[0]
known_face_encodings.append(face_encoding_person1)
known_face_names.append('Person1')

image_person2 = face_recognition.load_image_file('person2.jpg')
face_encoding_person2 = face_recognition.face_encodings(image_person2)[0]
known_face_encodings.append(face_encoding_person2)
known_face_names.append('Person2')

# 영상 장치 초기에 초기화
cap = cv2.VideoCapture(0)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.2) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # BGR 이미지를 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 존재하는 얼굴 감지
        results = face_detection.process(image_rgb)
        
        face_locations = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                top_left = (bbox[0], bbox[1])
                bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                face_locations.append((top_left[1], bottom_right[0], bottom_right[1], top_left[0]))

        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            face_names.append(name)
        
        # 결과 그리기
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # 얼굴의 사각형 그리기
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            # 이름 라벨 그리기
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        # 결과 출력
        cv2.imshow('Face Recognition', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
