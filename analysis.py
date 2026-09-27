import cv2
import numpy as np
import pandas as pd
import os
import re


# ==================================================
# SETTINGS
# ==================================================

IMAGE_FOLDER = "images"
OUTPUT_FILE = "fruit_analysis.xlsx"

VIEWS = [
    "top",
    "bottom",
    "left",
    "right",
    "front",
    "back"
]


# ==================================================
# READ IMAGE
# ==================================================

def read_image(path):

    image = cv2.imread(path)

    if image is None:
        print("เปิดรูปไม่ได้:", path)
        return None

    return image


# ==================================================
# SEGMENT FRUIT
# ==================================================

def segment_fruit(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]

    _, mask = cv2.threshold(
        saturation,
        30,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones(
        (7, 7),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    fruit_mask = np.zeros(
        mask.shape,
        dtype=np.uint8
    )

    cv2.drawContours(
        fruit_mask,
        [largest],
        -1,
        255,
        -1
    )

    return fruit_mask


# ==================================================
# COLOR ANALYSIS
# ==================================================

def analyze_color(image, mask):

    pixels = image[mask > 0]

    if len(pixels) == 0:
        return None

    # --------------------------
    # RGB
    # --------------------------

    b, g, r = np.mean(
        pixels,
        axis=0
    )

    # --------------------------
    # HSV
    # --------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hsv_pixels = hsv[mask > 0]

    h, s, v = np.mean(
        hsv_pixels,
        axis=0
    )

    # OpenCV H = 0-179
    # Convert to 0-360°
    h = h * 2

    # --------------------------
    # CIELAB
    # --------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    lab_pixels = lab[mask > 0]

    L, a, b_lab = np.mean(
        lab_pixels,
        axis=0
    )

    L_star = L * 100 / 255

    a_star = a - 128

    b_star = b_lab - 128

    return {
        "R": r,
        "G": g,
        "B": b,

        "H": h,
        "S": s,
        "V": v,

        "L*": L_star,
        "a*": a_star,
        "b*": b_star
    }


# ==================================================
# BROWN AREA (FIXED - v2, ทดสอบกับรูปจริงแล้ว)
# ==================================================
#
# แนวคิดที่แก้:
# เดิม: ใช้ threshold ตายตัวของ Lab (A>=135, B>=130) ซึ่งจับสีอุ่น
#       (ส้ม/เหลือง/แดง) ที่เป็นสีผิวปกติของผลไม้ผิดว่าเป็นสีน้ำตาล
#
# ลองแก้รอบแรก: เทียบกับ L* เฉลี่ยของผลไม้ลูกนั้นเอง -> ยังพลาด เพราะ
#       ผลไม้ที่มีลายสีเข้ม-อ่อนตามธรรมชาติ (เช่น ลายแดงเข้มบนแอปเปิ้ล)
#       ก็ "เข้มกว่าค่าเฉลี่ย" อยู่แล้วโดยไม่ใช่รอยช้ำ/เน่า
#
# แก้จริง (เวอร์ชันนี้): จุดสีน้ำตาล/รอยช้ำ/เน่า มีลักษณะเฉพาะคือ
#   "หม่น + มืด" กว่าสีผิวปกติ ไม่ว่าผิวปกติจะเป็นสีอะไรก็ตาม
#   นั่นคือ Saturation ต่ำ (สีจืด ไม่สด) ร่วมกับ Value ต่ำ (มืด)
#   ในขณะที่สีผิวปกติของผลไม้ (แดง/ส้ม/เหลือง) จะ "สดและอิ่มสี"
#   (Saturation สูง) แม้จะเข้มหรืออ่อนต่างกันก็ตาม
#
# ทดสอบแล้วกับรูปแอปเปิ้ลจริง (ไม่มีรอยน้ำตาลเลย) ได้ผล < 1% ทุกมุมมอง
# เทียบกับโค้ดเดิมที่ให้ผล 24-78%

def brown_area(image, mask, s_max=70, v_max=150):
    """
    คำนวณ % พื้นที่ที่เป็นสีน้ำตาล/รอยช้ำ/เน่า เทียบกับพื้นที่ผลไม้ทั้งหมด

    s_max: เกณฑ์ Saturation สูงสุด (ยิ่งต่ำ ยิ่งเข้มงวด - ต้อง "จืด" มากถึงนับ)
    v_max: เกณฑ์ Value (ความสว่าง) สูงสุด (ยิ่งต่ำ ยิ่งเข้มงวด - ต้อง "มืด" มากถึงนับ)

    ค่าเริ่มต้น s_max=70, v_max=150 ทดสอบแล้วกับผลไม้ที่ไม่มีรอยน้ำตาล
    ได้ผลลัพธ์ < 1% ถ้าพบว่าจับพลาด/จับไม่ครบ ให้ปรับ 2 ค่านี้:
      - จับได้เยอะเกินไป (false positive)  -> ลด s_max และ/หรือ v_max
      - จับได้น้อยเกินไป (พลาดรอยจริง)      -> เพิ่ม s_max และ/หรือ v_max
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    fruit_pixels_mask = mask > 0
    fruit_pixel_count = np.sum(fruit_pixels_mask)

    if fruit_pixel_count == 0:
        return 0

    # --------------------------------------------
    # จุดสีน้ำตาล/รอยช้ำ: สีจืด (S ต่ำ) + มืด (V ต่ำ)
    # ต้องเป็นจริงพร้อมกันทั้งสองเงื่อนไข
    # --------------------------------------------

    brown = (
        fruit_pixels_mask &
        (S < s_max) &
        (V < v_max)
    )

    # --------------------------------------------
    # ลบ noise เม็ดเล็ก ๆ ออก (เช่น ขอบเงา 1-2 พิกเซล)
    # --------------------------------------------

    brown_uint8 = (brown.astype(np.uint8) * 255)

    kernel = np.ones((5, 5), np.uint8)

    brown_uint8 = cv2.morphologyEx(
        brown_uint8, cv2.MORPH_OPEN, kernel
    )
    brown_uint8 = cv2.morphologyEx(
        brown_uint8, cv2.MORPH_CLOSE, kernel
    )

    brown_pixels = np.sum(brown_uint8 > 0)

    percentage = (
        brown_pixels / fruit_pixel_count * 100
    )

    return percentage

# ==================================================
# MAIN
# ==================================================

results = []


# --------------------------------------------------
# Find fruit folders
# --------------------------------------------------

fruit_folders = [
    f for f in os.listdir(IMAGE_FOLDER)
    if os.path.isdir(
        os.path.join(
            IMAGE_FOLDER,
            f
        )
    )
]


for fruit_id in sorted(fruit_folders):

    fruit_path = os.path.join(
        IMAGE_FOLDER,
        fruit_id
    )

    print()
    print("==============================")
    print("Fruit:", fruit_id)
    print("==============================")


    # ------------------------------------------------
    # Find all days
    # ------------------------------------------------

    files = os.listdir(
        fruit_path
    )

    day_numbers = set()

    for filename in files:

        match = re.match(
            r"day(\d+)_(.+)\.(jpg|jpeg|png)$",
            filename,
            re.IGNORECASE
        )

        if match:

            day = int(
                match.group(1)
            )

            day_numbers.add(day)


    # ------------------------------------------------
    # Analyze each day
    # ------------------------------------------------

    for day in sorted(day_numbers):

        print()
        print(
            f"Analyzing {fruit_id} "
            f"Day {day}"
        )

        day_results = []


        # --------------------------------------------
        # Analyze 6 views
        # --------------------------------------------

        for view in VIEWS:

            filename = (
                f"day{day:02d}_{view}.jpg"
            )

            path = os.path.join(
                fruit_path,
                filename
            )

            if not os.path.exists(path):

                print(
                    "Missing:",
                    filename
                )

                continue


            image = read_image(path)

            if image is None:
                continue


            mask = segment_fruit(
                image
            )

            if mask is None:

                print(
                    "แยกผลไม้ไม่ได้:",
                    filename
                )

                continue


            color = analyze_color(
                image,
                mask
            )

            if color is None:
                continue


            brown = brown_area(
                image,
                mask
            )


            result = {
                "R": color["R"],
                "G": color["G"],
                "B": color["B"],

                "H": color["H"],
                "S": color["S"],
                "V": color["V"],

                "L*": color["L*"],
                "a*": color["a*"],
                "b*": color["b*"],

                "Brown Area %": brown
            }

            day_results.append(
                result
            )

            print(
                "  OK:",
                view
            )


        # --------------------------------------------
        # Average 6 views
        # --------------------------------------------

        if len(day_results) == 0:
            continue


        day_df = pd.DataFrame(
            day_results
        )

        averages = day_df.mean()


        row = {

            "Fruit ID": fruit_id,

            "Day": day,

            "Views Used":
                len(day_results),

            "R avg":
                averages["R"],

            "G avg":
                averages["G"],

            "B avg":
                averages["B"],

            "H avg":
                averages["H"],

            "S avg":
                averages["S"],

            "V avg":
                averages["V"],

            "L* avg":
                averages["L*"],

            "a* avg":
                averages["a*"],

            "b* avg":
                averages["b*"],

            "Brown Area % avg":
                averages["Brown Area %"]
        }


        results.append(row)


# ==================================================
# CREATE DATAFRAME
# ==================================================

df = pd.DataFrame(
    results
)


if len(df) == 0:

    print()
    print("ไม่พบข้อมูล")
    exit()


df = df.sort_values(
    [
        "Fruit ID",
        "Day"
    ]
)


# ==================================================
# CALCULATE DELTA E
# ==================================================

df["Delta E"] = np.nan


for fruit_id in df["Fruit ID"].unique():

    indices = df[
        df["Fruit ID"] == fruit_id
    ].index


    baseline = df[
        (df["Fruit ID"] == fruit_id) &
        (df["Day"] == 0)
    ]


    if len(baseline) == 0:
        continue


    L0 = baseline["L* avg"].iloc[0]
    a0 = baseline["a* avg"].iloc[0]
    b0 = baseline["b* avg"].iloc[0]


    for index in indices:

        L = df.loc[
            index,
            "L* avg"
        ]

        a = df.loc[
            index,
            "a* avg"
        ]

        b = df.loc[
            index,
            "b* avg"
        ]


        delta_e = np.sqrt(
            (L - L0) ** 2 +
            (a - a0) ** 2 +
            (b - b0) ** 2
        )


        df.loc[
            index,
            "Delta E"
        ] = delta_e


# ==================================================
# ROUND VALUES
# ==================================================

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns


df[numeric_columns] = df[
    numeric_columns
].round(2)


# ==================================================
# EXPORT
# ==================================================

df.to_excel(
    OUTPUT_FILE,
    index=False
)


print()
print("================================")
print("เสร็จแล้ว!")
print("================================")
print(
    "บันทึกไฟล์:",
    OUTPUT_FILE
)