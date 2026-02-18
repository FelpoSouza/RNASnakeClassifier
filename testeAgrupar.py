import os
import shutil
from pathlib import Path

def agrupar_pastas(diretorio_base):
    # lista todas as pastas dentro do diretório base
    for item in os.listdir(diretorio_base):
        caminho_item = os.path.join(diretorio_base, item)
        
        if os.path.isdir(caminho_item) and "_" in item:
            # pega a parte antes do "_" -> ex: "Nome1"
            prefixo = item.split("_")[0]
            pasta_destino = os.path.join(diretorio_base, prefixo)

            # cria a pasta principal se não existir
            os.makedirs(pasta_destino, exist_ok=True)

            # move a subpasta para dentro da pasta principal
            destino = os.path.join(pasta_destino, item)
            if not os.path.exists(destino):
                shutil.move(caminho_item, pasta_destino)
            else:
                print(f"A pasta {destino} já existe, não movida.")


def contar_pastas(diretorio_base):
    base = Path(diretorio_base)
    externas = 0
    internas = 0

    for item in base.iterdir():
        if item.is_dir():
            if "_" in item.name:      # internas
                internas += 1
            else:                     # externas
                externas += 1

    return externas, internas

# exemplo de uso
if __name__ == "__main__":
    caminho = "/mnt/c/Users/Enzo/Documents/Faculdade/RNA/DATASET_RESIZE"  # coloque o caminho da pasta base
    agrupar_pastas(caminho)
    print("Agrupamento concluído!")

