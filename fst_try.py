import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# 장치 설정 (GPU 사용 가능 시 GPU, 없으면 CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MTCNN: 여러 얼굴 감지를 위해 keep_all=True
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)
# FaceNet 모델 (InceptionResnetV1) 로드
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 미리 등록한 얼굴 데이터베이스 (예시)
# 실제로는 각 사람의 기준 사진으로부터 임베딩을 계산해 저장해야 합니다.
# 여기서는 예시로 두 명의 임베딩을 임의의 값으로 채워놓았습니다.
known_faces = {
    "Alice": torch.randn(512).to(device),  # 실제 임베딩 벡터로 대체
    "Bob": torch.randn(512).to(device)
}

def match_face(face_embedding, known_faces, threshold=0.8):
    """
    입력 얼굴 임베딩과 등록된 known_faces의 임베딩을 cosine similarity로 비교.
    가장 유사도가 높은 얼굴을 반환하며, threshold 미만이면 "Unknown" 처리.
    """
    best_score = -1
    best_name = "Unknown"
    for name, known_embedding in known_faces.items():
        # 배치 차원을 추가해 cosine similarity 계산
        cos_sim = torch.nn.functional.cosine_similarity(face_embedding, known_embedding.unsqueeze(0)).item()
        if cos_sim > best_score and cos_sim > threshold:
            best_score = cos_sim
            best_name = name
    return best_name, best_score

# 웹캠 스트림 열기
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV BGR 이미지를 RGB로 변환 후 PIL 이미지로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # MTCNN을 통해 여러 얼굴의 bounding box와 얼굴 crop 이미지 얻기
    boxes, _ = mtcnn.detect(pil_img)
    faces = mtcnn(pil_img)  # keep_all=True이면 여러 얼굴의 crop 이미지를 반환

    if boxes is not None and faces is not None:
        for i, box in enumerate(boxes):
            # bounding box 좌표 정수형 변환
            box = [int(b) for b in box]
            # 해당 인덱스의 얼굴 crop 이미지 얻기 (tensor, shape: [3, 160, 160])
            if i < len(faces):
                face_img = faces[i]
                # 배치 차원 추가하여 임베딩 계산
                face_embedding = resnet(face_img.unsqueeze(0).to(device))
                # 등록된 얼굴과 매칭
                name, score = match_face(face_embedding, known_faces, threshold=0.8)
                
                # bounding box와 이름 표시
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({score:.2f})", (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Webcam Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
