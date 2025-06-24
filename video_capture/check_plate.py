import time

from video_capture.trusted_plates_util import remove_plate_from_file


def check_plate(plate):
    filename = "plates.txt"
    with open(filename, "r") as f:
        known_plates = [line.strip() for line in f]

    if plate in known_plates:
        print(f"✅ Zaufana tablica: {plate} – otwieranie bramy...")
        remove_plate_from_file(plate)
        time.sleep(5)
    else:
        print(f"❌ Tablica nieznana: {plate} – brama pozostaje zamknięta.")
        time.sleep(2)