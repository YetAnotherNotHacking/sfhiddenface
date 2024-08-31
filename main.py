import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def mosaic_blur(image, block_size):
    h, w, _ = image.shape
    mosaic = image.copy()
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            x_end = min(x + block_size, w)
            y_end = min(y + block_size, h)
            block = image[y:y_end, x:x_end]
            color = block.mean(axis=(0, 1))
            mosaic[y:y_end, x:x_end] = color
    return mosaic

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        mosaic_image = mosaic_blur(image_bgr, 10)
        mask_image = np.zeros_like(image_bgr)
        image_lines = np.zeros_like(image_bgr)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image_bgr.shape
                landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

                face_outline = [landmarks[i] for i in [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
                                                       365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
                                                       132, 93, 234, 127, 162, 21, 54, 103, 67, 109]]

                lip_top = landmarks[13]
                lip_bottom = landmarks[14]
                lip_left = landmarks[61]
                lip_right = landmarks[291]
                left_eye_top = landmarks[159]
                left_eye_bottom = landmarks[145]

                right_eye_top = landmarks[386]
                right_eye_bottom = landmarks[374]

                # Lip connections (form oval)
                cv2.line(image_lines, lip_top, landmarks[61], (0, 255, 0), 2)
                cv2.line(image_lines, landmarks[61], lip_bottom, (0, 255, 0), 2)
                cv2.line(image_lines, lip_bottom, landmarks[291], (0, 255, 0), 2)
                cv2.line(image_lines, landmarks[291], lip_top, (0, 255, 0), 2)

                cv2.polylines(image_lines, [np.array(face_outline)], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(image_lines, lip_top, 3, (0, 0, 255), -1)
                cv2.circle(image_lines, lip_bottom, 3, (0, 0, 255), -1)
                cv2.circle(image_lines, lip_left, 3, (0, 0, 255), -1)
                cv2.circle(image_lines, lip_right, 3, (0, 0, 255), -1)
                cv2.circle(image_lines, left_eye_top, 3, (0, 0, 255), -1)
                cv2.circle(image_lines, left_eye_bottom, 3, (0, 0, 255), -1)
                cv2.circle(image_lines, right_eye_top, 3, (0, 0, 255), -1)
                cv2.circle(image_lines, right_eye_bottom, 3, (0, 0, 255), -1)


                cv2.fillPoly(mask_image, [np.array(face_outline)], color=(255, 255, 255))
                global pixelated_overlay
                pixelated_overlay = np.where(mask_image == (255, 255, 255), mosaic_image, image_bgr)

        def labelText(img, text, font, fontScale, color, thickness):
            textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
            textX = (img.shape[1] - textsize[0]) // 2
            textY = 30
            cv2.putText(img, text, (textX, textY), font, fontScale, color, thickness)

        top_left = image_lines
        top_right = mosaic_image
        bottom_left = np.zeros_like(image_bgr)
        bottom_left[mask_image == (255, 255, 255)] = image_bgr[mask_image == (255, 255, 255)]
        bottom_right = pixelated_overlay
        labelText(top_left, 'Face Outline', cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        labelText(top_right, 'Mosaic Blur', cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        labelText(bottom_left, 'Face Mask', cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        labelText(bottom_right, 'Pixelated Overlay', cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        top_row = np.hstack([top_left, top_right])
        bottom_row = np.hstack([bottom_left, bottom_right])
        final_output = np.vstack([top_row, bottom_row])

        cv2.imshow('Face Mesh Outputs', final_output)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
