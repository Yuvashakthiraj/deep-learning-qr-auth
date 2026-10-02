import qrcode
import os

# ---------------- CONFIG ----------------
SAVE_PATH = "../dataset/new"
TOTAL_QRS = 1   # 👈 change this number (500 / 1000 / 2000)
PREFIX = "PAYMENT"
# ----------------------------------------

os.makedirs(SAVE_PATH, exist_ok=True)

for i in range(1, TOTAL_QRS + 1):
    data = f"{PREFIX}_{str(i).zfill(5)}"  # PAYMENT_00001, PAYMENT_00002 ...

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(os.path.join(SAVE_PATH, f"{data}.png"))

print(f"{TOTAL_QRS} original QR codes generated successfully.")
