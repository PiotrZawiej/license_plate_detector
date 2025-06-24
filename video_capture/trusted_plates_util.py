
# function only for presentation

filename = "plates.txt"


def save_known_plates():
    plates = [
        "WE424LG",
        "KR845TM",
        "PO329XR",
        "ZS104PN",
        "DW553LU",
        "LU920AE",
    ]

    with open(filename, "w") as f:
        for plate in plates:
            f.write(plate + "\n")


def remove_plate_from_file(plate_to_remove):
    try:
        with open(filename, "r") as f:
            plates = [line.strip() for line in f if line.strip() != plate_to_remove]

        with open(filename, "w") as f:
            for plate in plates:
                f.write(plate + "\n")

        print("*** Usuwanie zaufanej tablicy z listy na potrzeby prezentacji... ***")
    except FileNotFoundError:
        print(f"❌ Plik '{filename}' nie istnieje.")



