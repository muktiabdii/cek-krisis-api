import csv
import re

def load_kamus_slang(path):
    kamus = {}
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            slang = row['kata_slang'].strip().lower()
            baku = row['kata_baku'].strip().lower()
            kamus[slang] = baku
    return kamus


def normalize(text, kamus):
    # Urutkan slang berdasarkan panjang karakter (agar frasa panjang diganti lebih dulu)
    sorted_slang = sorted(kamus.keys(), key=len, reverse=True)

    # Ganti satu per satu slang dengan kata baku
    for slang in sorted_slang:
        # Gunakan regex agar bisa mengganti frasa/kata secara tepat
        pattern = r'\b' + re.escape(slang) + r'\b'
        text = re.sub(pattern, kamus[slang], text, flags=re.IGNORECASE)

    return text
