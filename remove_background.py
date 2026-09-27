"""
สคริปต์ตัดพื้นหลังผลไม้ทั้งหมดในโฟลเดอร์ images/ แบบอัตโนมัติ (batch)
แล้วบันทึกทับไฟล์เดิม (ชื่อไฟล์เดิม)

ก่อนรัน ต้องติดตั้งไลบรารีก่อน (รันครั้งเดียวใน terminal):
    pip install rembg

วิธีใช้:
    python remove_background.py

คำเตือน:
- สคริปต์นี้จะ "เขียนทับไฟล์เดิม" จริง ๆ แนะนำให้ก๊อปปี้โฟลเดอร์ images/
  สำรองไว้ก่อนรันครั้งแรก เผื่อผลลัพธ์ไม่ถูกใจจะได้มีต้นฉบับกลับไปใช้
- ใช้เวลาต่อรูปประมาณ 1-3 วินาที ถ้ามีหลายร้อยรูปอาจใช้เวลาหลายนาที
- ไม่แก้ปัญหา "มือติดในเฟรม" เพราะ AI มองมือ+ผลไม้เป็นวัตถุเดียวกัน
  ถ้ารูปไหนมีมือถือผลไม้อยู่ ควรครอปมือออกเอง หรือถ่ายใหม่ก่อนรันสคริปต์นี้
"""

from rembg import remove, new_session
from PIL import Image
import os

# ==================================================
# SETTINGS
# ==================================================

IMAGE_FOLDER = "images"

# โมเดลที่ใช้: u2netp = เบา เร็ว (แนะนำสำหรับรูปพื้นหลังเรียบ/สตูดิโอ)
# ถ้าอยากได้ความแม่นยำสูงขึ้น (แต่ช้าลงมาก) เปลี่ยนเป็น "u2net" หรือ "isnet-general-use"
MODEL_NAME = "u2netp"

# ขนาดสูงสุดที่จะย่อรูปลงก่อนประมวลผล (พิกเซล) - ช่วยประหยัดเวลา/แรม
# ถ้าอยากได้ความละเอียดเต็ม ให้ตั้งเป็น None (จะช้าลงและกินแรมมากขึ้น)
MAX_SIZE = 1000

# นามสกุลไฟล์ที่จะประมวลผล
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ==================================================
# MAIN
# ==================================================

def remove_background_batch():

    session = new_session(MODEL_NAME)

    processed_count = 0
    error_count = 0

    # เดินลึกเข้าไปทุกโฟลเดอร์ย่อยของ images/ (แต่ละลูกผลไม้)
    for root, dirs, files in os.walk(IMAGE_FOLDER):

        for filename in sorted(files):

            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue

            path = os.path.join(root, filename)

            try:
                img = Image.open(path).convert("RGB")

                if MAX_SIZE:
                    img.thumbnail((MAX_SIZE, MAX_SIZE))

                # ตัดพื้นหลังออก -> ได้ภาพที่มีพื้นหลังโปร่งใส (RGBA)
                result = remove(img, session=session)

                # แปลงพื้นหลังโปร่งใส ให้เป็นพื้นขาวล้วน
                # (เพราะโค้ดวิเคราะห์สีเดิมอ่านไฟล์ JPG ธรรมดา ไม่รองรับโปร่งใส)
                white_bg = Image.new("RGB", result.size, (255, 255, 255))
                white_bg.paste(result, mask=result.split()[3])  # ใช้ alpha channel เป็น mask

                # บันทึกทับไฟล์เดิม (ชื่อเดิม นามสกุลเดิม)
                if filename.lower().endswith((".jpg", ".jpeg")):
                    white_bg.save(path, quality=95)
                else:
                    white_bg.save(path)

                processed_count += 1
                print(f"  OK: {path}")

            except Exception as e:
                error_count += 1
                print(f"  [ผิดพลาด] {path}: {e}")

    print()
    print("================================")
    print(f"เสร็จแล้ว! ประมวลผลสำเร็จ {processed_count} ไฟล์")
    if error_count > 0:
        print(f"มีปัญหา {error_count} ไฟล์ (ดูรายละเอียดด้านบน)")
    print("================================")


if __name__ == "__main__":
    remove_background_batch()