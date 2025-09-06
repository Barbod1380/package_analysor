# app.py
import os
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
    if not path or not os.path.isfile(path):
        return "_"
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            return line if line else "_"
    except Exception:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                line = f.readline().strip()
                return line if line else "_"
        except Exception:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                line = f.readline().strip()
                return line if line else "_"


def read_text_all(path: Optional[str]) -> str:
    if not path or not os.path.isfile(path):
        return "_"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return content if content else "_"
    except Exception:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                content = f.read().strip()
                return content if content else "_"
        except Exception:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                return content if content else "_"


def load_image(path: Optional[str], max_side: int = 900) -> Optional[Image.Image]:
    """Safely load an image. Returns a PIL Image or None. Avoids opening directories."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with Image.open(path) as im:
            img = im.convert("RGB")
            w, h = img.size
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)))
            return img.copy()  # detach from context manager
    except Exception:
        return None


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def find_zip_root_with_required_dirs(extracted_dir: str) -> Optional[str]:
    """
    Return a directory that contains ALL required subdirs.
    1) Try the top-level extracted_dir
    2) Search deeper for a dir that contains all required dirs
    """
    if all(os.path.isdir(os.path.join(extracted_dir, d)) for d in REQUIRED_DIRS):
        return extracted_dir
    for root, dirs, _files in os.walk(extracted_dir):
        if all(os.path.isdir(os.path.join(root, d)) for d in REQUIRED_DIRS):
            return root
    return None


def collect_images(images_dir: str) -> Dict[str, str]:
    """Recursively collect images, mapping filename stem (case-insensitive) to full path."""
    image_map: Dict[str, str] = {}
    if not os.path.isdir(images_dir):
        return image_map
    for dirpath, _dirs, files in os.walk(images_dir):
        for name in files:
            if name.lower().endswith(IMG_EXTS):
                p = os.path.join(dirpath, name)
                if os.path.isfile(p):
                    image_map[stem(p).lower()] = p
    return image_map


def find_by_stem_case_insensitive(folder: str, base: str) -> Optional[str]:
    """
    Find a file in 'folder' whose stem matches 'base', case-insensitive,
    regardless of extension case.
    """
    if not folder or not os.path.isdir(folder):
        return None
    base_low = base.lower()
    try:
        for name in os.listdir(folder):
            p = os.path.join(folder, name)
            if not os.path.isfile(p):
                continue
            # match by stem only
            if stem(name).lower() == base_low:
                return p
        # if not found at top level, walk deeper:
        for dirpath, _dirs, files in os.walk(folder):
            for name in files:
                p = os.path.join(dirpath, name)
                if os.path.isfile(p) and stem(name).lower() == base_low:
                    return p
    except PermissionError:
        # If the folder has restricted entries, just skip them.
        return None
    return None


# ----------------- Index building -----------------

def build_index(root: str) -> Dict:
    """
    Build index based on 'falses_normalized_rotated' as the reference set.
    For each base name A:
      - image: A.(ext)
      - postcode crop: same stem in 'postcode_img_preprocessed' (any ext)
      - receiver crop: same stem in 'receiver_img_preprocessed' (any ext)
      - postcode txt: 'read_postcode/A_postcode.txt' (first line)
      - words txt:    'read_words/A_words.txt' (entire content)
      - region txt:   'address_region_pred/A_words_region_by_addr.txt' (first line)
    """
    images_dir = os.path.join(root, FOLDER_IMAGES)
    pcode_img_dir = os.path.join(root, FOLDER_POSTCODE_IMG)
    recv_img_dir = os.path.join(root, FOLDER_RECEIVER_IMG)
    read_postcode_dir = os.path.join(root, FOLDER_READ_POSTCODE)
    read_words_dir = os.path.join(root, FOLDER_READ_WORDS)
    region_dir = os.path.join(root, FOLDER_REGION)

    image_map = collect_images(images_dir)
    keys = sorted(image_map.keys())

    entries = []
    for k_low in keys:
        # k_low is lowercase stem
        k_path = image_map.get(k_low)
        k = stem(k_path) if k_path else k_low  # keep original case (nice for header)

        # NOTE: For an image like 'my_file.png', 'k' would be 'my_file'.

        # EXPECTS: A file in 'postcode_img_preprocessed' with the same stem (e.g., 'my_file.jpg')
        postcode_img = find_by_stem_case_insensitive(pcode_img_dir, k)
        
        # EXPECTS: A file in 'receiver_img_preprocessed' with the same stem (e.g., 'my_file.bmp')
        receiver_img = find_by_stem_case_insensitive(recv_img_dir, k)

        # EXPECTS: A file in 'read_postcode' named exactly '<image_stem>_postcode.txt' (e.g., 'my_file_postcode.txt')
        postcode_txt = os.path.join(read_postcode_dir, f"{k}_postcode.txt")
        
        # EXPECTS: A file in 'read_words' named exactly '<image_stem>_words.txt' (e.g., 'my_file_words.txt')
        words_txt = os.path.join(read_words_dir, f"{k}_words.txt")

        # EXPECTS: A file in 'address_region_pred' named exactly '<image_stem>_words_region_by_addr.txt'
        # (e.g., 'my_file_words_region_by_addr.txt')
        region_txt = os.path.join(region_dir, f"{k}_words_region_by_addr.txt")

        entries.append({
            "key": k,
            "image": k_path,
            "postcode_img": postcode_img,
            "receiver_img": receiver_img,
            "postcode_txt": postcode_txt if os.path.isfile(postcode_txt) else None,
            "words_txt": words_txt if os.path.isfile(words_txt) else None,
            "region_txt": region_txt if os.path.isfile(region_txt) else None,
        })

    return {
        "root": root,
        "entries": entries,
        "n": len(entries),
        "i": 0,  # current index pointer
    }


# ----------------- Gradio callbacks -----------------

def show_current(state: dict, show_debug: bool):
    if not state or state.get("n", 0) == 0:
        return (None, None, None,
                "<div>No data.</div>", "<div>No data.</div>", "<div>No data.</div>",
                state, "No data.", "")

    i = state["i"]
    n = state["n"]
    e = state["entries"][i]

    main_img = load_image(e["image"])
    pc_img = load_image(e["postcode_img"])
    rc_img = load_image(e["receiver_img"])

    postcode = read_text_first_line(e["postcode_txt"])
    words = read_text_all(e["words_txt"])
    region = read_text_first_line(e["region_txt"])

    # Persian RTL HTML blocks
    words_html = f'<div dir="rtl" style="text-align:right; white-space:pre-wrap;">{words}</div>'
    postcode_html = f'<div dir="rtl" style="text-align:right;">{postcode}</div>'
    region_html = f'<div dir="rtl" style="text-align:right;">{region}</div>'

    header = f"{e['key']}  ({i+1}/{n})"

    debug_text = ""
    if show_debug:
        debug_text = (
            f"<pre style='white-space:pre-wrap'>"
            f"image:         {e['image']}\n"
            f"postcode_img:  {e['postcode_img']}\n"
            f"receiver_img:  {e['receiver_img']}\n"
            f"postcode_txt:  {e['postcode_txt']}\n"
            f"words_txt:     {e['words_txt']}\n"
            f"region_txt:    {e['region_txt']}\n"
            f"</pre>"
        )

    return main_img, pc_img, rc_img, words_html, postcode_html, region_html, state, header, debug_text


def load_zip_and_prepare(zip_file, show_debug: bool):
    """
    1) Extract ZIP to a temp dir
    2) Find the root that contains the 6 folders
    3) Build index and show first item
    """
    if zip_file is None:
        empty = (None, None, None, "", "", "", {}, "Please upload a ZIP first.", "")
        dropdown_update = gr.update(choices=[], value=None)
        status = "Please upload a ZIP."
        return (*empty, dropdown_update, status)

    zip_path = getattr(zip_file, "name", None) or str(zip_file)
    if not os.path.isfile(zip_path):
        empty = (None, None, None, "", "", "", {}, "Invalid ZIP file.", "")
        dropdown_update = gr.update(choices=[], value=None)
        status = "Invalid ZIP file."
        return (*empty, dropdown_update, status)

    tmpdir = tempfile.mkdtemp(prefix="dataset_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)

    root = find_zip_root_with_required_dirs(tmpdir)
    if root is None:
        empty = (None, None, None, "", "", "", {}, "Required folders not found in ZIP.", "")
        dropdown_update = gr.update(choices=[], value=None)
        status = f"Could not find all required folders: {', '.join(REQUIRED_DIRS)}"
        return (*empty, dropdown_update, status)

    state = build_index(root)
    state["tmpdir"] = tmpdir  # keep alive for session

    if state["n"] == 0:
        empty = (None, None, None, "", "", "", state, "No images found in falses_normalized_rotated.", "")
        dropdown_update = gr.update(choices=[], value=None)
        status = "No images found."
        return (*empty, dropdown_update, status)

    main_img, pc_img, rc_img, words_html, postcode_html, region_html, state, header, debug_text = show_current(state, show_debug)
    keys = [e["key"] for e in state["entries"]]
    dropdown_update = gr.update(choices=keys, value=keys[0] if keys else None)
    status = f"Loaded {state['n']} items."
    return main_img, pc_img, rc_img, words_html, postcode_html, region_html, state, header, debug_text, dropdown_update, status


def goto_prev(state: dict, show_debug: bool):
    if not state or state.get("n", 0) == 0:
        return (None, None, None, "", "", "", state, "No data.", "")
    state["i"] = (state["i"] - 1) % state["n"]
    return show_current(state, show_debug)


def goto_next(state: dict, show_debug: bool):
    if not state or state.get("n", 0) == 0:
        return (None, None, None, "", "", "", state, "No data.", "")
    state["i"] = (state["i"] + 1) % state["n"]
    return show_current(state, show_debug)


def goto_key(state: dict, key: str, show_debug: bool):
    if not state or state.get("n", 0) == 0:
        return (None, None, None, "", "", "", state, "No data.", "")
    if key:
        for idx, e in enumerate(state["entries"]):
            if e["key"] == key:
                state["i"] = idx
                break
    return show_current(state, show_debug)


def list_keys(state: dict):
    if not state or state.get("n", 0) == 0:
        return gr.update(choices=[], value=None)
    keys = [e["key"] for e in state["entries"]]
    return gr.update(choices=keys, value=keys[state["i"]] if 0 <= state["i"] < len(keys) else keys[0])


# ----------------- UI -----------------

with gr.Blocks(title="Postal Data Viewer") as demo:
    gr.Markdown("### Postal Data Viewer — upload one ZIP containing the 6 folders")

    with gr.Row():
        zip_input = gr.File(label="Upload ZIP with the 6 required folders", file_types=[".zip"])
        show_debug = gr.Checkbox(label="Show debug paths", value=False)
        load_btn = gr.Button("Load", variant="primary")

    status = gr.Markdown("", elem_id="status")

    with gr.Row():
        header = gr.Markdown("")
        key_dropdown = gr.Dropdown(choices=[], label="Jump to key", interactive=True)

    with gr.Row():
        main_img = gr.Image(label="Normalized image (A)", interactive=False, type="pil")
        pc_img   = gr.Image(label="Postcode box",         interactive=False, type="pil")
        rc_img   = gr.Image(label="Receiver box",         interactive=False, type="pil")

    with gr.Row():
        words_html    = gr.HTML(label="Words (Persian)")
        postcode_html = gr.HTML(label="Postcode (first line)")
        region_html   = gr.HTML(label="Region (from address)")

    debug_md = gr.Markdown("", visible=True)

    with gr.Row():
        prev_btn = gr.Button("⬅ Prev")
        next_btn = gr.Button("Next ➡")

    # hidden state
    state = gr.State({})

    # Wire up
    load_btn.click(
        load_zip_and_prepare,
        inputs=[zip_input, show_debug],
        outputs=[
            main_img, pc_img, rc_img,
            words_html, postcode_html, region_html,
            state, header, debug_md, key_dropdown, status
        ],
    )

    prev_btn.click(
        goto_prev,
        inputs=[state, show_debug],
        outputs=[main_img, pc_img, rc_img, words_html, postcode_html, region_html, state, header, debug_md],
    )

    next_btn.click(
        goto_next,
        inputs=[state, show_debug],
        outputs=[main_img, pc_img, rc_img, words_html, postcode_html, region_html, state, header, debug_md],
    )

    key_dropdown.change(
        goto_key,
        inputs=[state, key_dropdown, show_debug],
        outputs=[main_img, pc_img, rc_img, words_html, postcode_html, region_html, state, header, debug_md],
    )

if __name__ == "__main__":
    demo.launch()