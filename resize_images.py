#!/usr/bin/env python3
"""
resize_images.py

Percorre recursivamente um diretório de entrada, redimensiona/imputa todas as imagens e salva
no diretório de saída mantendo a estrutura de pastas. Permite escolher o tamanho alvo e o modo:

Modos disponíveis:
  - crop   : redimensiona preservando proporção e depois recorta (center-crop) para o quadrado alvo
  - pad    : redimensiona preservando proporção e adiciona padding para formar um quadrado
  - stretch: força o redimensionamento para o quadrado (distorce se necessário)

Exemplos:
  python resize_images.py --input ./imagens --output ./out --size 240 --mode crop
  python resize_images.py -i fotos -o fotos_240 -s 240 -m pad --overwrite

Requisitos:
  pip install pillow tqdm

Feito para uso geral — modifique conforme precisar.
"""

import argparse
import os
from pathlib import Path
from PIL import Image, ImageOps
import sys

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = img.size
    left = max(0, (w - target_w) // 2)
    upper = max(0, (h - target_h) // 2)
    right = left + target_w
    lower = upper + target_h
    return img.crop((left, upper, right, lower))


def pad_to_square(img: Image.Image, target: int, background=(0, 0, 0, 0)) -> Image.Image:
    # mantém alfa se existir
    mode = img.mode
    if mode in ("RGBA", "LA") or (mode == "P" and 'transparency' in img.info):
        new_mode = 'RGBA'
    else:
        new_mode = 'RGB'

    img = img.convert(new_mode)
    w, h = img.size
    new_img = Image.new(new_mode, (target, target), background)
    left = (target - w) // 2
    top = (target - h) // 2
    new_img.paste(img, (left, top), img if new_mode == 'RGBA' else None)
    return new_img


def process_image(in_path: Path, out_path: Path, size: int, mode: str):
    try:
        with Image.open(in_path) as img:
            # corrige orientação via EXIF
            img = ImageOps.exif_transpose(img)
            w, h = img.size

            if mode == 'stretch':
                new = img.resize((size, size), Image.LANCZOS)
            elif mode == 'crop':
                # primeiro redimensiona tal que a menor dimensão seja == size
                # assim garantindo que ao recortar a dimensão sobrará
                scale = max(size / w, size / h)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                resized = img.resize((new_w, new_h), Image.LANCZOS)
                new = center_crop(resized, size, size)
            elif mode == 'pad':
                scale = min(size / w, size / h)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                resized = img.resize((new_w, new_h), Image.LANCZOS)
                # define cor de fundo: transparente para imagens com alfa, branco para RGB
                bg = (255, 255, 255)
                if resized.mode in ('RGBA', 'LA'):
                    bg = (255, 255, 255, 0)
                new = pad_to_square(resized, size, background=bg)
            else:
                raise ValueError(f"Modo desconhecido: {mode}")

            # preserva o formato original (extensão)
            ensure_dir(out_path.parent)
            # ao salvar JPEG, remover alpha se existir
            fmt = img.format or out_path.suffix.replace('.', '').upper()
            save_kwargs = {}
            if fmt == 'JPEG' or out_path.suffix.lower() in ('.jpg', '.jpeg'):
                if new.mode in ('RGBA', 'LA'):
                    new = new.convert('RGB')
                save_kwargs['quality'] = 95

            new.save(out_path, **save_kwargs)
            return True, None
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Resize imagens recursivamente para um quadrado de tamanho fixo')
    parser.add_argument('-i', '--input', required=True, help='Diretório de entrada (será percorrido recursivamente)')
    parser.add_argument('-o', '--output', required=True, help='Diretório de saída (estrutura de pastas será preservada)')
    parser.add_argument('-s', '--size', required=True, type=int, help='Tamanho do lado do quadrado em pixels (ex: 240)')
    parser.add_argument('-m', '--mode', choices=['crop', 'pad', 'stretch'], default='crop', help='Modo de ajuste (default: crop)')
    parser.add_argument('--overwrite', action='store_true', help='Sobrescrever arquivos de saída existentes')
    parser.add_argument('--dry-run', action='store_true', help='Mostrar o que seria feito, sem salvar')
    parser.add_argument('--verbose', action='store_true', help='Imprime detalhes')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    size = args.size
    mode = args.mode

    if not input_dir.exists():
        print('Diretório de entrada não existe:', input_dir)
        sys.exit(1)

    total = 0
    succeeded = 0
    failed = 0
    failures = []

    # coleta todos os arquivos de imagem
    img_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in SUPPORTED_EXTS:
                img_files.append(p)

    for in_path in tqdm(img_files, desc='Processando'):
        rel = in_path.relative_to(input_dir)
        out_path = output_dir / rel
        # ajusta extensão se quiser forçar formato, mas por ora mantém

        if not args.overwrite and out_path.exists():
            if args.verbose:
                print('Pulando (já existe):', out_path)
            continue

        total += 1
        if args.dry_run:
            if args.verbose:
                print('Dry run:', in_path, '->', out_path)
            succeeded += 1
            continue

        ok, err = process_image(in_path, out_path, size, mode)
        if ok:
            succeeded += 1
            if args.verbose:
                print('OK:', in_path, '->', out_path)
        else:
            failed += 1
            failures.append((in_path, err))
            if args.verbose:
                print('ERRO:', in_path, err)

    print('\nResumo:')
    print('  Imagens encontradas:', len(img_files))
    print('  Processadas (tentadas):', total)
    print('  Sucesso:', succeeded)
    print('  Falhas:', failed)
    if failures:
        print('\nErros (amostra):')
        for p, e in failures[:10]:
            print(' -', p, ':', e)


if __name__ == '__main__':
    main()
