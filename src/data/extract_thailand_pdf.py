"""
Extract province-level maize data from OAE PDF tables.
Converts rai → ha and kg/rai → ton/ha.
Outputs standard format CSV.
"""
import re
import logging
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd
import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
RAI_TO_HA = 1.0 / 6.25  # 1 rai = 0.16 ha
KG_RAI_TO_TON_HA = 0.00625  # 1 kg/rai = 6.25 kg/ha = 0.00625 ton/ha

# 77 Thai provinces in canonical Thai script
_THAI_PROVINCES = [
    "กระบี่", "กรุงเทพมหานคร", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
    "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท", "ชัยภูมิ",
    "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง", "ตราด", "ตาก",
    "นครนายก", "นครปฐม", "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์",
    "นนทบุรี", "นราธิวาส", "น่าน", "บึงกาฬ", "บุรีรัมย์",
    "ปทุมธานี", "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา",
    "พะเยา", "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี",
    "เพชรบูรณ์", "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร",
    "แม่ฮ่องสอน", "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง",
    "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน", "เลย", "ศรีสะเกษ",
    "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร",
    "สระแก้ว", "สระบุรี", "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี",
    "สุรินทร์", "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี",
    "อุตรดิตถ์", "อุทัยธานี", "อุบลราชธานี",
]


def _buddhist_to_calendar(be_year: int) -> int:
    """Convert Buddhist year to calendar year (planting year)."""
    return be_year - 543


def _normalize_thai(text: str) -> str:
    """Normalize Thai text from PDF extraction artifacts.

    Handles: PUA characters, decomposed chars, zero-width chars,
    encoding variants across different PDF generators.
    """
    if not text:
        return text

    # Remove BOM, zero-width, and other invisible markers
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\u2060"):
        text = text.replace(ch, "")

    # Replace common PUA (Private Use Area) Thai character variants
    pua_map = {
        "\uf701": "ั",
        "\uf702": "็",
        "\uf703": "ิ",
        "\uf704": "่",
        "\uf705": "้",
        "\uf70a": "่",
        "\uf70b": "ข",
        "\uf70e": "์",
        "\uf712": "ำ",
    }
    for pua, canonical in pua_map.items():
        text = text.replace(pua, canonical)

    # NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Decomposed sara am: nikhahit (ํ U+0E4D) + sara aa (า U+0E32) → sara am (ำ U+0E33)
    text = text.replace("\u0E4D\u0E32", "\u0E33")

    # Remove ALL whitespace within Thai text (Thai doesn't use intra-word spaces)
    result = []
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.isspace():
            prev_thai = i > 0 and "\u0E00" <= chars[i - 1] <= "\u0E7F"
            next_thai = i < len(chars) - 1 and "\u0E00" <= chars[i + 1] <= "\u0E7F"
            if prev_thai and next_thai:
                continue
        result.append(ch)
    text = "".join(result)

    return text.strip()


def _match_province(name: str) -> Optional[str]:
    """Match extracted name to canonical Thai province name using fuzzy matching."""
    from difflib import SequenceMatcher

    name_norm = _normalize_thai(name)

    # Exact match after normalization
    if name_norm in _THAI_PROVINCES:
        return name_norm

    # Fuzzy match
    best_ratio = 0.0
    best_match = None
    for prov in _THAI_PROVINCES:
        ratio = SequenceMatcher(None, name_norm, prov).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = prov

    if best_ratio >= 0.60:  # Lower threshold for Thai PDF encoding variations
        return best_match

    return None


def _is_aggregate(text: str) -> bool:
    """Check if row is a national/regional aggregate to skip."""
    text_norm = _normalize_thai(text)
    # Check for aggregate keywords (loose match due to PDF encoding issues)
    agg_patterns = ["รวม", "ภาค", "ประเทศ", "ประเทศไทย"]
    for pat in agg_patterns:
        pat_norm = _normalize_thai(pat)
        if pat_norm in text_norm:
            return True
    # Also check with fuzzy against known aggregates
    from difflib import SequenceMatcher
    known_aggs = [
        "รวมทั้งประเทศ", "ภาคเหนือ", "ภาคกลาง", "ภาคใต้",
        "ภาคตะวันออกเฉียงเหนือ", "ภาคตะวันออก",
    ]
    for agg in known_aggs:
        if SequenceMatcher(None, text_norm, agg).ratio() >= 0.7:
            return True
    return False


def _parse_number(val: str) -> Optional[float]:
    """Parse Thai-formatted number string (comma-separated) to float."""
    if val is None:
        return None
    val = str(val).strip().replace(",", "")
    if val in ("", "-", "0") or not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _extract_crop_year_buddhist(text: str) -> int:
    """Extract Buddhist year (ปเีพาะปลูก 25XX/XX) from PDF first-page text."""
    match = re.search(r"ปีเพาะปลูก\s*(\d{4})", text)
    if not match:
        match = re.search(r"ป[\u0E00-\u0E7F]*เพาะปลูก\s*(\d{4})", text)
    if match:
        return int(match.group(1))
    # Fallback: try first 4-digit number in 25xx range
    matches = re.findall(r"(25\d{2})", text)
    for m in matches:
        yr = int(m)
        if 2550 <= yr <= 2570:
            return yr
    raise ValueError(f"Cannot find Buddhist year in text: {text[:200]}")


def extract_oae_pdf(pdf_path: Path) -> pd.DataFrame:
    """Extract province-level maize data from a single OAE PDF.

    Returns DataFrame with columns:
        region_name, year, crop_year_thai, planted_ha, harvested_ha,
        production_ton, yield_ton_ha
    """
    logger.info(f"Processing: {pdf_path.name}")

    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    rows.append(row)

    # Parse crop year
    be_year = _extract_crop_year_buddhist(full_text)
    calendar_year = _buddhist_to_calendar(be_year)
    crop_year_thai = f"{be_year}/{be_year - 2500 + 1}"
    logger.info(f"  Crop year: {crop_year_thai} (Buddhist) → {calendar_year} (calendar)")

    # Parse data rows
    data = []
    unmatched = set()
    for row in rows:
        if not row or not row[0]:
            continue
        raw_name = str(row[0]).strip()
        if not raw_name or raw_name.startswith("จังหวัด/อ") or raw_name.startswith("ภาค/") or raw_name.startswith("ประเทศ"):
            continue
        if _is_aggregate(raw_name):
            continue

        # Match to canonical province name
        canonical = _match_province(raw_name)
        if canonical is None:
            unmatched.add(raw_name)
            continue

        # Column layout (all cycles = รวมรุ่น):
        # 0=province, 1=planted_rai, 2=harvested_rai, 3=production_ton,
        # 4=yield_planted_kg_rai, 5=yield_harvested_kg_rai
        planted_rai = _parse_number(row[1])
        harvested_rai = _parse_number(row[2])
        production_ton = _parse_number(row[3])

        if planted_rai is None or planted_rai == 0:
            continue

        planted_ha = planted_rai * RAI_TO_HA
        harvested_ha = harvested_rai * RAI_TO_HA if harvested_rai else None
        yield_ton_ha = production_ton / harvested_ha if production_ton and harvested_ha else None

        data.append(
            {
                "region_name": canonical,
                "year": calendar_year,
                "crop_year_thai": crop_year_thai,
                "planted_ha": round(planted_ha, 2),
                "harvested_ha": round(harvested_ha, 2) if harvested_ha else None,
                "production_ton": round(production_ton, 2) if production_ton else None,
                "yield_ton_ha": round(yield_ton_ha, 4) if yield_ton_ha else None,
            }
        )

    if unmatched:
        logger.warning(f"  Unmatched rows ({len(unmatched)}): {sorted(unmatched)}")

    df = pd.DataFrame(data)
    # Some provinces appear twice (as province and with sub-district rows);
    # keep only the row with the largest planted area (true province total)
    df = df.sort_values("planted_ha", ascending=False).drop_duplicates(
        subset=["region_name", "year"], keep="first"
    )
    df = df.sort_values(["year", "region_name"]).reset_index(drop=True)

    logger.info(f"  Extracted {len(df)} province rows ({df['region_name'].nunique()} unique)")
    return df


def main():
    base_dir = Path("/Users/alisulas/ClaudeQ/thesis_maize/data/Manually Download/Thailand")
    output_dir = Path("/Users/alisulas/ClaudeQ/thesis_maize/data/processed/thailand")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = {
        "2021.pdf": "crop_year_2564_65",
        "corn 2024.pdf": "crop_year_2565_66",
        "Corn 2023.pdf": "crop_year_2566_67",
    }

    all_data = []
    for filename, label in pdf_files.items():
        pdf_path = base_dir / filename
        if not pdf_path.exists():
            logger.warning(f"  Skipping missing: {pdf_path}")
            continue
        df = extract_oae_pdf(pdf_path)
        all_data.append(df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(["year", "region_name"]).reset_index(drop=True)

        # Save combined
        out_path = output_dir / "thailand_province_yield_2021_2023.csv"
        combined.to_csv(out_path, index=False)
        logger.info(f"Saved: {out_path}")
        logger.info(f"Total rows: {len(combined)}")
        logger.info(f"Years: {sorted(combined['year'].unique())}")
        logger.info(f"Provinces: {combined['region_name'].nunique()}")
        logger.info(f"Yield range: {combined['yield_ton_ha'].dropna().min():.2f} – {combined['yield_ton_ha'].dropna().max():.2f} ton/ha")

        print("\n=== SAMPLE ===")
        print(combined.head(10).to_string())
        print(f"\n=== STATS ===")
        print(combined[["year", "planted_ha", "harvested_ha", "production_ton", "yield_ton_ha"]].describe())
    else:
        logger.error("No data extracted!")


if __name__ == "__main__":
    main()
