import cv2
import numpy as np

# ── 1) 클래스 이름과 DNN 모델 로드 ─────────────────────────────
CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle",
           "bus","car","cat","chair","cow","diningtable","dog",
           "horse","motorbike","person","pottedplant","sheep",
           "sofa","train","tvmonitor"]

net = cv2.dnn.readNetFromCaffe(
    "models/MobileNetSSD_deploy.prototxt",
    "models/MobileNetSSD_deploy.caffemodel"
)

# ── 2) IOU / 중심 거리 계산 ───────────────────────────────────
def compute_iou(box1, box2):
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2
    xi1,yi1 = max(x1,x2), max(y1,y2)
    xi2,yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = w1*h1 + w2*h2 - inter
    return inter/union if union>0 else 0

def center_distance(box1, box2):
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2
    c1 = np.array([x1+w1/2, y1+h1/2])
    c2 = np.array([x2+w2/2, y2+h2/2])
    return np.linalg.norm(c1-c2)

# ── 3) NMS 로 중복 박스 제거 ──────────────────────────────────
def remove_duplicate_boxes_with_nms(boxes, confidences, iou_threshold=0.3):
    if not boxes:
        return []
    # boxes: [[x,y,w,h],...], confidences: [c1,c2,...]
    idxs = cv2.dnn.NMSBoxes(
        boxes, confidences,
        score_threshold=0.0,      # 이미 conf 필터링 했으니 0
        nms_threshold=iou_threshold
    )
    # 반환이 [[i],[j],...] 꼴일 수 있어 평탄화
    idxs = [i[0] if hasattr(i,"__iter__") else i for i in idxs]
    return [boxes[i] for i in idxs]


def detect_blue_players(frame):
    blob = cv2.dnn.blobFromImage(
        frame, 0.007843,
        (300,300),
        (127.5,127.5,127.5)
    )
    net.setInput(blob)
    det = net.forward()

    cand_boxes, confidences = [], []
    H, W = frame.shape[:2]

    for i in range(det.shape[2]):
        conf = float(det[0,0,i,2])
        cls  = int(det[0,0,i,1])
        if conf > 0.10 and CLASSES[cls]=="person":
            x1,y1,x2,y2 = (det[0,0,i,3:7] * np.array([W,H,W,H])).astype(int)
            if x2<=x1 or y2<=y1: 
                continue

            w, h = x2-x1, y2-y1
            # 작은 박스도 허용 (너무 작으면 잡아내기 어려우므로 소폭 필터)
            if w < 5 or h < 10:
                continue

            roi = frame[y1:y2, x1:x2]
            # 색상 분류 안정화를 위해 블러 + HSV 변환
            hsv = cv2.cvtColor(cv2.GaussianBlur(roi,(7,7),0),
                               cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv,
                               (100,100,50),   # H:100~130, S:100~255, V:50~255
                               (130,255,255))
            blue_ratio = cv2.countNonZero(mask) / (w*h)

    
            if blue_ratio > 0.02:
                cand_boxes.append([x1,y1,w,h])
                confidences.append(conf)

    return remove_duplicate_boxes_with_nms(
               cand_boxes, confidences,
               iou_threshold=0.3
           )

# ── 5) 위치 연결선 그리기 ────────────────────────────────────
def draw_position_lines(frame, boxes, ids):
    centers = [(x+w//2, y+h//2, tid)
               for (x,y,w,h), tid in zip(boxes,ids)]
    centers.sort(key=lambda c:c[0])
    for i in range(len(centers)-1):
        pt1 = (centers[i][0], centers[i][1])
        pt2 = (centers[i+1][0], centers[i+1][1])
        cv2.line(frame, pt1, pt2, (0,255,0), 2)

# ── 6) 메인 루프: 트래커 생성·갱신·표시 ───────────────────────
cap = cv2.VideoCapture("Soccer4.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
wait = int(1000/fps)

trackers, track_ids, colors = [], [], []
fail_counts = {}
next_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_blue_players(frame)
    assigned = [False]*len(detections)

    updated_trk, updated_ids, updated_cols = [], [], []

    # 기존 트래커 업데이트 & 매칭
    for trk, tid, col in zip(trackers, track_ids, colors):
        ok, old_box = trk.update(frame)
        if not ok:
            fail_counts[tid] = fail_counts.get(tid,0)+1
            if fail_counts[tid] >= 5:
                continue
        else:
            fail_counts[tid] = 0

        old_box = [int(v) for v in old_box]
        best_iou, best_j = 0, -1

        for j, new_box in enumerate(detections):
            if assigned[j]: continue
            iou  = compute_iou(old_box, new_box)
            dist = center_distance(old_box, new_box)
            if iou>0.4 and dist<50 and iou>best_iou:
                best_iou, best_j = iou, j

        if best_j>=0:
            # 매칭된 거로 트래커 재초기화
            trk = cv2.legacy.TrackerCSRT_create()
            trk.init(frame, tuple(detections[best_j]))
            assigned[best_j] = True

        updated_trk.append(trk)
        updated_ids.append(tid)
        updated_cols.append(col)

    # 신규 detection → 새 트래커
    for j, box in enumerate(detections):
        if assigned[j]:
            continue
        trk = cv2.legacy.TrackerCSRT_create()
        trk.init(frame, tuple(box))
        updated_trk.append(trk)
        updated_ids.append(next_id)
        updated_cols.append((255,0,255))
        next_id += 1

    # 다음 프레임용 필터링 & 그리기
    trackers, track_ids, colors = [], [], []
    final_boxes = []

    for trk, tid, col in zip(updated_trk, updated_ids, updated_cols):
        ok, box = trk.update(frame)
        if not ok:
            continue
        x,y,w,h = [int(v) for v in box]
        if not (10<w<300 and 10<h<300 and 0<=x<frame.shape[1] and 0<=y<frame.shape[0]):
            continue

        trackers.append(trk)
        track_ids.append(tid)
        colors.append(col)
        final_boxes.append((x,y,w,h))
        cv2.rectangle(frame, (x,y), (x+w,y+h), col, 2)
        cv2.putText(frame, f"People:{tid}", (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # 연결선 그리기
    draw_position_lines(frame, final_boxes, track_ids)

    cv2.imshow("Team Position Line (Blue Only)", frame)
    if cv2.waitKey(wait) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
