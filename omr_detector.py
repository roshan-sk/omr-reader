import cv2
import numpy as np
import json


def get_contours(edged):
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 100000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            return approx

    return None


def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)

    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]

    return new_points


def warp_image(image, points):
    points = reorder(points)

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [800, 0], [0, 1200], [800, 1200]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, matrix, (800, 1200))

    return warped

def manual_crop_column(col_img):
    h, w = col_img.shape[:2]

    TOP_MARGIN = 0
    BOTTOM_MARGIN = 0
    LEFT_MARGIN = 50
    RIGHT_MARGIN = 0

    cropped = col_img[TOP_MARGIN:h - BOTTOM_MARGIN,
                      LEFT_MARGIN:w - RIGHT_MARGIN]

    
    top_crop = int(h * 0.03)
    bubble_area = cropped[top_crop:h, :]

    return bubble_area


def preprocess_and_warp(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found. Check the file path!")
        return None
    
    image = cv2.resize(image, (800, 1200))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    contour = get_contours(edged)

    if contour is None:
        # print("Could not detect the OMR sheet. Using full image.")
        return image

    area = cv2.contourArea(contour)


    if area < 300000:
        # print("Detected area is too small. Image may not be clear. Using full image.")
        return image

    return warp_image(image, contour)


def split_into_columns(bubble_only, num_cols = 5):
    h, w = bubble_only.shape[:2]

    
    col_width = w // num_cols

    columns = []

    for i in range(num_cols):
        x1 = i * col_width
        x2 = (i + 1) * col_width

        col = bubble_only[:, x1:x2]
        columns.append(col)

    return columns


def get_threshold(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 2
    )

    return thresh


def detect_rows(thresh):
    row_sum = np.sum(thresh, axis=1)
    row_sum = row_sum / np.max(row_sum)

    rows = []
    in_row = False

    for i in range(len(row_sum)):
        if row_sum[i] > 0.2 and not in_row:
            start = i
            in_row = True
        elif row_sum[i] < 0.2 and in_row:
            end = i
            rows.append((start, end))
            in_row = False

    merged = []
    min_gap = 5

    for row in rows:
        if not merged:
            merged.append(list(row))
            continue

        prev = merged[-1]
        gap = row[0] - prev[1]

        if 0 < gap < min_gap:
            prev[1] = row[1]
        else:
            merged.append(list(row))

    final_rows = []
    for (s, e) in merged:
        if (e - s) > 15:
            final_rows.append((s, e))

    return sorted(final_rows, key=lambda x: x[0])


def detect_answers(thresh, rows):
    answers = []

    for q_idx, (start, end) in enumerate(rows):
        row_img = thresh[start:end, :]

        h, w = row_img.shape
        option_width = w // 4

        scores = []

        for i in range(4):
            x1 = i * option_width
            x2 = (i + 1) * option_width

            option = row_img[:, x1:x2]
            total = cv2.countNonZero(option)
            scores.append(total)

        answers.append(np.argmax(scores))

    return answers


def process_column(col_img, c_indx):
    thresh = get_threshold(col_img)
    rows = detect_rows(thresh)
    answers = detect_answers(thresh, rows)
    return answers


def main():
    warped = preprocess_and_warp("sample_omrs/omr_sheet2.jpg")
    # cv2.imshow("wrapped_show", warped)
    # cv2.imwrite("wrapped.png", warped)
    columns = split_into_columns(warped)

    all_answers = []

    for i, col in enumerate(columns):

        # cv2.imshow(f"CR {i+1}", cv2.resize(col, (200, 600)))
        # cv2.imwrite(f"before_crop-{i+1}.png", col)
        cleaned_col = manual_crop_column(col)

        # cv2.imshow(f"R {i+1}", cv2.resize(cleaned_col, (200, 600)))
        # cv2.imwrite(f"after_crop-{i+1}.png", cleaned_col)

        answers = process_column(cleaned_col, i+1)
        all_answers.extend(answers)

    options = ['1', '2', '3', '4']
    final_answers = []
    for i, a in enumerate(all_answers, start=1):
        if a >= 0:
            final_answers.append(f"{i}-{options[a]}")
        else:
            final_answers.append(f"{i}-unknown")

    print("Total questions detected:", len(final_answers))
    print("Final Answers:")
    print(final_answers)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

