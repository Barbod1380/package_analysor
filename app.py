# app.py
import os
import io
import re
import glob
import shutil
import zipfile
import tempfile
from typing import Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

# ---- Expected folder names inside the ZIP ----
FOLDER_IMAGES = "falses_normalized_rotated"
FOLDER_POSTCODE_IMG = "postcode_img_preprocessed"
FOLDER_RECEIVER_IMG = "receiver_img_preprocessed"
FOLDER_READ_POSTCODE = "read_postcode"
FOLDER_READ_WORDS = "read_words"
FOLDER_REGION = "address_region_pred"

REQUIRED_DIRS = [
    FOLDER_IMAGES,
    FOLDER_POSTCODE_IMG,
    FOLDER_RECEIVER_IMG,
    FOLDER_READ_POSTCODE,
    FOLDER_READ_WORDS,
    FOLDER_REGION,
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ----------------- Small helpers -----------------

def read_text_first_line(path: Optional[str]) -> str:
    if not path or not os.path.exists(path):
        return "_"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readline().strip() or "_"
    except Exception:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readline().strip() or "_"

def read_text_all(path: Optional[str]) -> str:
    if not path or not os.path.exists(path):
        return "_"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return (f.read().strip() or "_")
    except Exception:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return (f.read().strip() or "_")

def load_image(path: Optional[str], max_side: int = 900) -> Optional[Image.Image]:
    if not path or not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return img

def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def find_zip_root_with_required_dirs(extracted_dir: str) -> Optional[str]:
    """
    Return a directory that contains ALL required subdirs.
    1) Try the top-level extracted_dir
    2) Search a couple levels deep for a dir that contains all required dirs
    """
    # direct check
    if all(os.path.isdir(os.path.join(extracted_dir, d)) for d in REQUIRED_DIRS):
        return extracted_dir

    # walk to find candidate that contains all
    for root, dirs, _files in os.walk(extracted_dir):
        if all(os.path.isdir(os.path.join(root, d)) for d in REQUIRED_DIRS):
            return root
    return None

def find_any_with_same_stem(folder: str, base: str, exts: Tuple[str, ...]) -> Optional[str]:
    """Return first match of base with any of the exts in a folder."""
    for ext in exts:
        p = os.path.join(folder, base + ext)
        if os.path.exists(p):
            return p
    return None

# ----------------- Index building -----------------

def build_index(root: str) -> Dict:
    """
    Build index based on the 'falses_normalized_rotated' folder.
    For each base name A:
      - image: A.(ext)
      - postcode crop: same stem in postcode_img_preprocessed (any ext)
      - receiver crop: same stem in receiver_img_preprocessed (any ext)
      - postcode txt: read_postcode/A_postcode.txt
      - words txt:    read_words/A_words.txt
      - region txt:   address_region_pred/A_words_region_by_addr.txt
    """
    paths = {}
    images_dir = os.path.join(root, FOLDER_IMAGES)
    pcode_img_dir = os.path.join(root, FOLDER_POSTCODE_IMG)
    recv_img_dir = os.path.join(root, FOLDER_RECEIVER_IMG)
    read_postcode_dir = os.path.join(root, FOLDER_READ_POSTCODE)
    read_words_dir = os.path.join(root, FOLDER_READ_WORDS)
    region_dir = os.path.join(root, FOLDER_REGION)

    # collect primary images
    image_map: Dict[str, str] = {}
    for ext in IMG_EXTS:
        for p in glob.glob(os.path.join(images_dir, f"**/*{ext}"), recursive=True):
            image_map[stem(p)] = p

    keys = sorted(image_map.keys())

    entries = []
    for k in keys:
        # crops
        postcode_img = find_any_with_same_stem(pcode_img_dir, k, IMG_EXTS)
        receiver_img = find_any_with_same_stem(recv_img_dir, k, IMG_EXTS)
        # texts
        postcode_txt = os.path.join(read_postcode_dir, f"{k}_postcode.txt")
        words_txt = os.path.join(read_words_dir, f"{k}_words.txt")
        region_txt = os.path.join(region_dir, f"{k}_words_region_by_addr.txt")

        entries.append({
            "key": k,
            "image": image_map.get(k),
            "postcode_img": postcode_img,
            "receiver_img": receiver_img,
            "postcode_txt": postcode_txt if os.path.exists(postcode_txt) else None,
            "words_txt": words_txt if os.path.exists(words_txt) else None,
            "region_txt": region_txt if os.path.exists(region_txt) else None,
        })

    return {
        "root": root,
        "entries": entries,
        "n": len(entries),
        "i": 0,  # current index pointer
    }

# ----------------- Gradio callbacks -----------------

def load_zip_and_prepare(zip_file) -> Tuple[str, str, str, str, str, str, dict, str]:
    """
    1) Extract ZIP to a temp dir
    2) Find the root that contains the 6 folders
    3) Build index and show first item
    """
    if zip_file is None:
        return "", "", "", "", "", "", {}, "Please upload a ZIP."

    # extract to temp
    tmpdir = tempfile.mkdtemp(prefix="dataset_")
    with zipfile.ZipFile(zip_file.name, "r") as zf:
        zf.extractall(tmpdir)

    # find a root that has the required dirs
    root = find_zip_root_with_required_dirs(tmpdir)
    if root is None:
        return "", "", "", "", "", "", {}, (
            "Could not find all required folders inside the ZIP.\n"
            f"Expected: {', '.join(REQUIRED_DIRS)}"
        )

    state = build_index(root)
    if state["n"] == 0:
        return "", "", "", "", "", "", state, "No images found in falses_normalized_rotated."

    # show first
    return show_current(state)

def show_current(state: dict):
    if not state or state.get("n", 0) == 0:
        return "", "", "", "", "", "", state, "No data."

    i = state["i"]
    n = state["n"]
    e = state["entries"][i]

    # images
    main_img = load_image(e["image"])
    pc_img = load_image(e["postcode_img"])
    rc_img = load_image(e["receiver_img"])

    # texts
    postcode = read_text_first_line(e["postcode_txt"])
    words = read_text_all(e["words_txt"])
    region = read_text_first_line(e["region_txt"])

    # convert PIL -> displayable (Gradio Image accepts PIL directly)
    # Compose Persian/RTL display using HTML blocks:
    words_html = f"""
    <div dir="rtl" style="text-align:right; white-space:pre-wrap;">
      {words if words != '_' else '_'}
    </div>
    """

    postcode_html = f"""
    <div dir="rtl" style="text-align:right;">
      {postcode if postcode != '_' else '_'}
    </div>
    """

    region_html = f"""
    <div dir="rtl" style="text-align:right;">
      {region if region != '_' else '_'}
    </div>
    """

    header = f"{e['key']}  ({i+1}/{n})"

    return (main_img, pc_img, rc_img, words_html, postcode_html, region_html, state, header)

def goto_prev(state: dict):
    if not state or state.get("n", 0) == 0:
        return "", "", "", "", "", "", state, "No data."
    state["i"] = (state["i"] - 1) % state["n"]
    return show_current(state)

def goto_next(state: dict):
    if not state or state.get("n", 0) == 0:
        return "", "", "", "", "", "", state, "No data."
    state["i"] = (state["i"] + 1) % state["n"]
    return show_current(state)

def goto_key(state: dict, key: str):
    if not state or state.get("n", 0) == 0:
        return "", "", "", "", "", "", state, "No data."
    if not key:
        return show_current(state)
    # find index of this key
    for idx, e in enumerate(state["entries"]):
        if e["key"] == key:
            state["i"] = idx
            break
    return show_current(state)

def list_keys(state: dict) -> List[str]:
    if not state or state.get("n", 0) == 0:
        return []
    return [e["key"] for e in state["entries"]]

# ----------------- UI -----------------

with gr.Blocks(title="Postal Data Viewer") as demo:
    gr.Markdown("### Postal Data Viewer — one ZIP with 6 folders")

    with gr.Row():
        zip_input = gr.File(label="Upload ZIP with the 6 folders", file_types=[".zip"])
        load_btn = gr.Button("Load", variant="primary")

    status = gr.Markdown("", elem_id="status")

    # Header row: key and jump-to
    with gr.Row():
        header = gr.Markdown("")
        key_dropdown = gr.Dropdown(choices=[], label="Jump to key", interactive=True)

    with gr.Row():
        main_img = gr.Image(label="Normalized image (A)", interactive=False)
        pc_img = gr.Image(label="Postcode box", interactive=False)
        rc_img = gr.Image(label="Receiver box", interactive=False)

    with gr.Row():
        words_html = gr.HTML(label="Words (Persian)")
        postcode_html = gr.HTML(label="Postcode (first line)")
        region_html = gr.HTML(label="Region (from address)")

    with gr.Row():
        prev_btn = gr.Button("⬅ Prev")
        next_btn = gr.Button("Next ➡")

    # hidden state
    state = gr.State({})

    # wiring
    load_btn.click(
        load_zip_and_prepare,
        inputs=[zip_input],
        outputs=[main_img, pc_img, rc_img, words_html, postcode_html, region_html, state, header],
    ).then(
        list_keys, inputs=[state], outputs=[key_dropdown]
    ).then(
        lambda: "Loaded.", None, status
    )

    prev_btn.click(
        goto_prev,
        inputs=[state],
        outputs=[main_img, pc_img, rc_img, words_html, postcode_html, region_html, state, header],
    )

    next_btn.click(
        goto_next,
        inputs=[state],
        outputs=[main_img, pc_img, rc_img, words_html, postcode_html, region_html, state, header],
    )

    key_dropdown.change(
        goto_key,
        inputs=[state, key_dropdown],
        outputs=[main_img, pc_img, rc_img, words_html, postcode_html, region_html, state, header],
    )

if __name__ == "__main__":
    demo.launch()
