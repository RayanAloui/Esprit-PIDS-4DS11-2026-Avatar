import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import requests
from dotenv import load_dotenv


_DATA_DIR = Path(__file__).resolve().parent / "data"
_DEFAULT_CSV_PATH = _DATA_DIR / "vital_products.csv"


def _safe_filename(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value).strip())
    clean = clean.strip("_")
    return clean[:80] if clean else "product"


def _build_short_prompt(row: pd.Series) -> str:
    """Shorter prompt to avoid URL length issues"""
    name = str(row.get("name", "")).strip()
    categories = str(row.get("categories", "")).strip()
    forme = str(row.get("forme", "")).strip()
    
    return f"pharmaceutical marketing flyer, {name}, {categories}, {forme}, French text, professional healthcare design"


def _try_pollinations_v2(prompt: str, output_path: Path) -> Tuple[bool, str]:
    """Try Pollinations.ai with simpler endpoint"""
    try:
        url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
        params = {
            "width": 768,
            "height": 1024,
            "model": "turbo",
            "nologo": "true"
        }
        
        response = requests.get(url, params=params, timeout=90)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True, "pollinations_turbo"
    except Exception as e:
        return False, f"pollinations: {str(e)[:50]}"


def _try_pollinations_flux(prompt: str, output_path: Path) -> Tuple[bool, str]:
    """Try Pollinations.ai with Flux model (higher quality)"""
    try:
        url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
        params = {
            "width": 768,
            "height": 1024,
            "model": "flux",
            "nologo": "true"
        }
        
        response = requests.get(url, params=params, timeout=90)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True, "pollinations_flux"
    except Exception as e:
        return False, f"pollinations_flux: {str(e)[:50]}"


def _try_segmind(prompt: str, output_path: Path) -> Tuple[bool, str]:
    """Try Segmind free API"""
    try:
        response = requests.post(
            "https://api.segmind.com/v1/sd1.5-txt2img",
            json={
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, watermark",
                "samples": 1,
                "width": 768,
                "height": 1024,
                "steps": 20,
            },
            timeout=60
        )
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True, "segmind_sd15"
        return False, f"segmind: {response.status_code}"
    except Exception as e:
        return False, f"segmind: {str(e)[:50]}"


def _try_prodia(prompt: str, output_path: Path) -> Tuple[bool, str]:
    """Try Prodia free API"""
    try:
        gen_response = requests.post(
            "https://api.prodia.com/v1/sd/generate",
            json={
                "prompt": prompt,
                "model": "sd_xl_base_1.0.safetensors",
                "negative_prompt": "blurry, low quality",
                "steps": 20,
                "cfg_scale": 7,
                "width": 768,
                "height": 1024,
            },
            timeout=30
        )
        
        if gen_response.status_code != 200:
            return False, f"prodia gen failed: {gen_response.status_code}"
        
        job = gen_response.json()
        job_id = job.get("job")
        
        for _ in range(20):
            time.sleep(3)
            status_response = requests.get(
                f"https://api.prodia.com/v1/job/{job_id}",
                timeout=10
            )
            
            if status_response.status_code == 200:
                status = status_response.json()
                if status.get("status") == "succeeded":
                    image_url = status.get("imageUrl")
                    img_response = requests.get(image_url, timeout=30)
                    with open(output_path, 'wb') as f:
                        f.write(img_response.content)
                    return True, "prodia_sdxl"
        
        return False, "prodia: timeout"
    except Exception as e:
        return False, f"prodia: {str(e)[:50]}"


def _try_huggingface(prompt: str, output_path: Path) -> Tuple[bool, str]:
    """Try Hugging Face Inference API"""
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
            json={"inputs": prompt},
            timeout=60
        )
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True, "huggingface_sdxl"
        return False, f"huggingface: {response.status_code}"
    except Exception as e:
        return False, f"huggingface: {str(e)[:50]}"


def _try_deepai(prompt: str, output_path: Path, api_key: Optional[str]) -> Tuple[bool, str]:
    """Try DeepAI (has free tier with API key)"""
    if not api_key:
        return False, "deepai: no API key"
    
    try:
        response = requests.post(
            "https://api.deepai.org/api/text2img",
            data={'text': prompt},
            headers={'api-key': api_key},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            image_url = result.get("output_url")
            if image_url:
                img_response = requests.get(image_url, timeout=30)
                with open(output_path, 'wb') as f:
                    f.write(img_response.content)
                return True, "deepai"
        return False, f"deepai: {response.status_code}"
    except Exception as e:
        return False, f"deepai: {str(e)[:50]}"


def generate_flyers_from_csv(
    csv_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model: str = None,
    limit: Optional[int] = None,
) -> int:
    load_dotenv()
    
    csv_file = Path(csv_path) if csv_path else _DEFAULT_CSV_PATH
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    out_dir = Path(output_dir) if output_dir else (Path(__file__).resolve().parent / "generated_flyers")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file).fillna("")
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")

    if limit is not None and limit > 0:
        df = df.head(limit)

    deepai_key = os.getenv("DEEPAI_API_KEY", "").strip()

    print(f"[info] Trying ALL services for quality comparison")
    print(f"[info] CSV: {csv_file}")
    print(f"[info] Output dir: {out_dir}")
    print(f"[info] Get free DeepAI key at: https://deepai.org/machine-learning-model/text2img")
    print()

    total_generated = 0

    for idx, row in df.iterrows():
        product_name = str(row.get("name", f"product_{idx + 1}")).strip() or f"product_{idx + 1}"
        prompt = _build_short_prompt(row)
        safe_name = _safe_filename(product_name)

        print(f"{'='*70}")
        print(f"[{idx + 1}] Product: {product_name}")
        print(f"{'='*70}")
        
        # All generators to try
        generators = [
            ("Pollinations Turbo", lambda p: _try_pollinations_v2(prompt, p)),
            ("Pollinations Flux", lambda p: _try_pollinations_flux(prompt, p)),
            ("Hugging Face SDXL", lambda p: _try_huggingface(prompt, p)),
            ("Segmind SD1.5", lambda p: _try_segmind(prompt, p)),
            ("Prodia SDXL", lambda p: _try_prodia(prompt, p)),
            ("DeepAI", lambda p: _try_deepai(prompt, p, deepai_key)),
        ]
        
        successful_models: List[str] = []
        
        for service_name, generator_func in generators:
            # Create unique filename with model name
            # Will be overwritten by the generator with actual model name
            temp_path = out_dir / f"{idx + 1:03d}_{safe_name}_temp.png"
            
            print(f"  → Trying {service_name:25s}...", end=" ", flush=True)
            
            success, model_name = generator_func(temp_path)
            
            if success:
                # Rename file to include model name
                final_path = out_dir / f"{idx + 1:03d}_{safe_name}_{model_name}.png"
                if temp_path.exists():
                    temp_path.rename(final_path)
                    print(f"✓ SUCCESS → {final_path.name}")
                    successful_models.append(model_name)
                    total_generated += 1
                else:
                    print(f"✗ File not created")
            else:
                print(f"✗ {model_name}")
                # Clean up temp file if it exists
                if temp_path.exists():
                    temp_path.unlink()
            
            # Small delay between service attempts
            time.sleep(1.5)
        
        if successful_models:
            print(f"\n  ✓ Generated {len(successful_models)} versions: {', '.join(successful_models)}")
        else:
            print(f"\n  ✗ ALL SERVICES FAILED for: {product_name}")
        
        # Delay between products
        if idx < len(df) - 1:
            print()
            time.sleep(2)

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total products: {len(df)}")
    print(f"Total images generated: {total_generated}")
    print(f"Check {out_dir} to compare quality across different models")
    print(f"{'='*70}")
    
    return total_generated