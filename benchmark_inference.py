from ultralytics import YOLO
from pathlib import Path
import numpy as np
import csv

# =========================================================
# SETTINGS
# =========================================================
BASE_DIR = Path("/mnt/wd_mlai/Jessie_BT22")
IMAGE_DIR = BASE_DIR / "AI_FUMPE/images"

IMG_SIZE = 512
DEVICE = 0
WARMUP_RUNS = 10
SIZES = ["n", "s", "m", "l"]

OUTPUT_CSV = BASE_DIR / "inference_benchmark_results.csv"

# =========================================================
# BUILD MODEL PATHS (YOLO11n-l and YOLO26n-l)
# =========================================================
MODELS = {}

for size in SIZES:
    MODELS[f"YOLO11{size}"] = (
        BASE_DIR
        / "batch_16"
        / "yolo11"
        / "runs"
        / "segment"
        / f"11{size}"
        / "weights"
        / "best.pt"
    )

    MODELS[f"YOLO26{size}"] = (
        BASE_DIR
        / "batch_16"
        / "yolo26"
        / "runs"
        / "segment"
        / f"26{size}"
        / "weights"
        / "best.pt"
    )

# =========================================================
# LOAD ALL BENCHMARK IMAGES
# =========================================================
images = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [
        ".png", ".jpg", ".jpeg",
        ".bmp", ".tif", ".tiff"
    ]
])

if len(images) == 0:
    raise RuntimeError(f"No images found in: {IMAGE_DIR}")

print("=" * 70)
print("INFERENCE BENCHMARK")
print("=" * 70)
print(f"Total benchmark images : {len(images)}")
print(f"Image size             : {IMG_SIZE}")
print("Inference batch        : 1")
print(f"GPU device             : {DEVICE}")
print(f"Warm-up runs           : {WARMUP_RUNS}")
print(f"Models                 : {len(MODELS)}")
print("=" * 70)

# =========================================================
# RESULTS STORAGE
# =========================================================
all_results = []

# =========================================================
# BENCHMARK EACH MODEL
# =========================================================
for model_name, model_path in MODELS.items():

    print("\n" + "=" * 70)
    print(f"Benchmarking: {model_name}")
    print(f"Model path  : {model_path}")
    print("=" * 70)

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        continue

    model = YOLO(str(model_path))

    # -----------------------------------------------------
    # WARM-UP (not included in benchmark result)
    # -----------------------------------------------------
    print(f"Warm-up {WARMUP_RUNS} runs...")

    for _ in range(WARMUP_RUNS):
        model.predict(
            source=str(images[0]),
            imgsz=IMG_SIZE,
            device=DEVICE,
            batch=1,
            verbose=False
        )

    # -----------------------------------------------------
    # INFERENCE BENCHMARK
    # -----------------------------------------------------
    inference_times = []

    print("Starting benchmark...")

    for i, image_path in enumerate(images, start=1):

        results = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            device=DEVICE,
            batch=1,
            verbose=False
        )

        inference_ms = float(results[0].speed["inference"])
        inference_times.append(inference_ms)

        if i % 100 == 0 or i == len(images):
            print(f"Processed {i}/{len(images)} images")

    # -----------------------------------------------------
    # CALCULATE RESULTS
    # -----------------------------------------------------
    mean_ms = float(np.mean(inference_times))
    median_ms = float(np.median(inference_times))
    std_ms = float(np.std(inference_times))
    min_ms = float(np.min(inference_times))
    max_ms = float(np.max(inference_times))
    fps = 1000 / mean_ms

    all_results.append({
        "model": model_name,
        "images_tested": len(inference_times),
        "mean_ms_per_image": mean_ms,
        "median_ms": median_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "approx_fps": fps,
    })

    print("\nRESULT")
    print("-" * 50)
    print(f"Model               : {model_name}")
    print(f"Images tested       : {len(inference_times)}")
    print(f"Mean inference time : {mean_ms:.2f} ms/image")
    print(f"Median              : {median_ms:.2f} ms/image")
    print(f"Std                 : {std_ms:.2f} ms")
    print(f"Min                 : {min_ms:.2f} ms")
    print(f"Max                 : {max_ms:.2f} ms")
    print(f"Approx. FPS         : {fps:.2f}")
    print("-" * 50)

# =========================================================
# SAVE FINAL RESULTS TO CSV
# =========================================================
if all_results:
    fieldnames = [
        "model",
        "images_tested",
        "mean_ms_per_image",
        "median_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "approx_fps",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for row in all_results:
        print(
            f"{row['model']:10s} | "
            f"{row['mean_ms_per_image']:.2f} ms/image | "
            f"{row['approx_fps']:.2f} FPS"
        )

    print("\nSaved results to:")
    print(OUTPUT_CSV)
else:
    print("No benchmark results were generated.")
