import os
import random
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm

def add_gaussian_noise(img, sigma=10):
    """
    Adiciona ruído gaussiano à imagem.
    A imagem é convertida para numpy, adicionado ruído e reconvertida.
    """
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = arr + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def augment_small_classes(base_path, min_images=9):
    """
    Aumenta o número de imagens em classes pequenas (pastas com poucas imagens),
    gerando versões rotacionadas (90°, 180°, 270°) das imagens existentes.
    Mantém o formato original (JPEG ou PNG), evita duplicação e converte RGBA para RGB.
    """
    #mesmo que não use dir, é necessário estar aqui
    for root, dirs, files in os.walk(base_path):
        #identifica imagem como encerrada por .png e etc
        image_files = [
            f for f in files 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        #se não tem imagens, não tem como gerar mais
        if len(image_files) == 0:
            continue
        
        # ignora classes já suficientemente grandes
        if len(image_files) >= min_images:
            continue
        
        print(f"Classe pequena detectada em '{root}' ({len(image_files)} imagens)")
        
        for f in image_files:
            # evita processar imagens já rotacionadas, pois essas tem _rot no nome
            if "_rot" in f:
                continue
            #anexa nome da imagem ao caminho para acessá-la
            img_path = os.path.join(root, f)
            img = Image.open(img_path)

            # garantir que não há transparência
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            #adicionar extensão do arquivo no fim, replica o que já tinha, padrão é .jpg
            name, ext = os.path.splitext(f)
            new_ext = ext if ext.lower() in ['.jpg', '.jpeg', '.png'] else '.jpg'

            # gerar novas imagens rotacionadas
            for angle in [90, 180, 270]:
                img_rot = img.rotate(angle)
                #se voltou transparencia, remove A
                if img_rot.mode == 'RGBA':
                    img_rot = img_rot.convert('RGB')
                #nome novo tem _rot e o angulo
                new_name = f"{name}_rot{angle}{new_ext}"
                new_path = os.path.join(root, new_name)

                # se já existir, pula
                if os.path.exists(new_path):
                    continue

                img_rot.save(new_path, quality=95)
        #conta numero novo de imagens
        new_count = len([
            f for f in os.listdir(root)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        print(f"→ Agora a classe '{os.path.basename(root)}' tem {new_count} imagens.\n")

def balance_dataset(source_dir, output_dir):
    """
    Cria um dataset balanceado mantendo a estrutura:
        genero/
            especie/
                *.jpg

    Regras:
      - média total de imagens por espécie
      - target = media_imgs/2 + media_imgs/(1 + n_especies_genero)
      - poucas imagens → aumentar com ruído gaussiano
      - muitas imagens → subamostrar
      - salva tudo em output_dir
    """

    # ============================================================
    # 1) Coletar estatísticas
    # ============================================================

    especies_info = []  # lista de dicts: {genero, especie, paths}
    generos_dict = {}   # genero → lista de especies

    print("Lendo estrutura do dataset...")
    #Aqui a estrutura já está pronta, deve ser replicada
    for genero in os.listdir(source_dir):
        genero_path = os.path.join(source_dir, genero)
        if not os.path.isdir(genero_path):
            continue
        #associar um gênero específico com suas especie em um path
        generos_dict[genero] = []
        for especie in os.listdir(genero_path):
            especie_path = os.path.join(genero_path, especie)
            if not os.path.isdir(especie_path):
                continue
            #assocair a imagens da especie a ela via path
            imgs = [
                os.path.join(especie_path, f)
                for f in os.listdir(especie_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            #adicionar a espécie no gênero
            generos_dict[genero].append(especie)
            #associar, à espécie, o gênero a que pertence e suas imagens
            especies_info.append({
                "genero": genero,
                "especie": especie,
                "imgs": imgs
            })

    # Média geral
    total_imgs = sum(len(e["imgs"]) for e in especies_info)
    n_especies_total = len(especies_info)
    media_imgs = total_imgs / n_especies_total

    print(f"\n Média global de imagens por espécie: {media_imgs:.2f}")

    # ============================================================
    # 2) Criar diretório de saída
    # ============================================================
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 3) Processar cada espécie
    # ============================================================
    print("\n Balanceando dataset...\n")
    
    

    for info in tqdm(especies_info):
        # extrair informações coletadas no passo 1 de uma especie
        genero = info["genero"]
        especie = info["especie"]
        imgs = info["imgs"]
        #numero de imagens da espécie
        n_imgs = len(imgs)
        #número de espécies pertencentes a um gênero
        n_especies_genero = len(generos_dict[genero])

        # ---------- cálculo do número alvo ----------
        target = int(media_imgs/2 + media_imgs/(1 + n_especies_genero))
        target = max(target, 1)  # segurança

        out_dir_especie = os.path.join(output_dir, genero, especie)
        os.makedirs(out_dir_especie, exist_ok=True)

        # ============================================================
        # CASO 1: poucas imagens → aumentar com ruído
        # ============================================================
        if n_imgs < target:
            faltam = target - n_imgs

            # Copiar todas as imagens originais primeiro
            for path in imgs:
                shutil.copy(path, os.path.join(out_dir_especie, os.path.basename(path)))

            # Criar imagens sintéticas
            for i in range(faltam):
                base_img_path = random.choice(imgs)
                img = Image.open(base_img_path).convert("RGB")

                noisy = add_gaussian_noise(img)
                #adiciona _gauss_(numero atual da imagem gerada) ao nome, força .jpg
                new_name = f"{os.path.splitext(os.path.basename(base_img_path))[0]}_gauss_{i}.jpg"
                new_path = os.path.join(out_dir_especie, new_name)
                noisy.save(new_path, quality=95)

        # ============================================================
        # CASO 2: muitas imagens → subamostrar
        # ============================================================
        elif n_imgs > target:
            chosen = random.sample(imgs, target)
            for path in chosen:
                shutil.copy(path, os.path.join(out_dir_especie, os.path.basename(path)))

        # ============================================================
        # CASO 3: igual ao target → só copiar
        # ============================================================
        else:
            for path in imgs:
                shutil.copy(path, os.path.join(out_dir_especie, os.path.basename(path)))

    print("\n Dataset balanceado criado com sucesso!")
    print(f" Novo dataset salvo em: {output_dir}\n")

def main():
    SOURCE_DIR = "/mnt/c/Users/Enzo/Documents/Faculdade/RNA/DATASET_REDUZIDO"
    TARGET_DIR = "/mnt/c/Users/Enzo/Documents/Faculdade/RNA/DATASET_REDUZIDO_BALANCEADO"
    augment_small_classes(SOURCE_DIR, min_images=9)
    balance_dataset(SOURCE_DIR, TARGET_DIR)

if __name__ == "__main__":
    main()