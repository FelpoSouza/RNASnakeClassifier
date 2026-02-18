import os
import torch
import torch.nn as nn
from torch import amp
#from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import random

def salvar_matriz_confusao(y_true, y_pred, encoder, resumo, tipo="GENERO"):
    """
    Gera e salva matrizes de confusão binárias (1x1) para as classes do resumo F1.
    """
    for _, row in resumo.iterrows():
        #obter nome da classe com base no id
        idx_classe = int(row["classe"])
        classe_nome_real = encoder.classes_[idx_classe]

        # cria vetores binários: 1 = da classe, 0 = não da classe
        y_true_bin = np.array(y_true) == idx_classe
        y_pred_bin = np.array(y_pred) == idx_classe

        #gera matriz confusão, com a classe alvo X resto
        cm = confusion_matrix(y_true_bin, y_pred_bin)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Outro", classe_nome_real])

        #gerar imagem
        fig, ax = plt.subplots(figsize=(9, 9))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Matriz de Confusão ({tipo}) — {row['tipo']} F1\nClasse: {classe_nome_real}")

        save_name = f"matriz_confusao_{tipo.lower()}_{row['tipo']}_{classe_nome_real}.png"
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        plt.close(fig)

        print(f"Matriz de confusão salva: {save_name}")


def resumo_f1(y_true, y_pred, nome=""):
    # gera o report como dicionário
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # ignora as chaves "accuracy", "macro avg" e "weighted avg", 
    # geraremos 2 e temos os gráficosm isso é o suficiente
    df = pd.DataFrame(report).T
    df = df[~df.index.isin(["accuracy", "macro avg", "weighted avg"])]

    # obtém a coluna de F1
    df["f1-score"] = df["f1-score"].astype(float)
    
    # ordena por F1
    df_sorted = df.sort_values("f1-score")

    # identifica menor, mediano e maior
    pior = df_sorted.iloc[0]
    mediana = df_sorted.iloc[len(df_sorted)//2]
    melhor = df_sorted.iloc[-1]

    #gera um resumo, associando as 3 classes vistas
    resumo = pd.DataFrame({
        "classe": [df_sorted.index[0], df_sorted.index[len(df_sorted)//2], df_sorted.index[-1]],
        "precision": [pior["precision"], mediana["precision"], melhor["precision"]],
        "recall": [pior["recall"], mediana["recall"], melhor["recall"]],
        "f1-score": [pior["f1-score"], mediana["f1-score"], melhor["f1-score"]],
        "tipo": ["pior", "mediana", "melhor"]
    })

    print(f"\nResumo F1 — {nome}")
    print(resumo.to_string(index=False))
    return resumo

# =====================================================
# 1. Dataset personalizado
# =====================================================

class SnakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        #define o caminho raiz do diretorio com gêneros
        self.root_dir = root_dir
        #qual o transform que será aplicado em imagens
        self.transform = transform
        # guarda-se amostras
        self.samples = []

        # Lê estrutura: gênero -> espécie -> imagem
        for genero in os.listdir(root_dir):
            #pega caminho para acessar pasta de gênero
            genero_path = os.path.join(root_dir, genero)
            #ignora se não é diretório
            if not os.path.isdir(genero_path):
                continue
            #acessar diretorio de especies dentro do gênero
            for especie in os.listdir(genero_path):
                #pega caminho para acessar pasta de espécie
                especie_path = os.path.join(genero_path, especie)
                if not os.path.isdir(especie_path):
                    continue
                #acessar imagens da pasta de espécies
                for gname in os.listdir(especie_path):
                    if gname.lower().endswith((".jpg", ".jpeg", ".png")):
                        #em samples, adiciona o caminho até a imagem
                        # além de seu diretório genero e especie
                        self.samples.append((
                            os.path.join(especie_path, gname),
                            genero,
                            especie
                        ))

        # Gera codificador de gênero e espécie
        self.genero_encoder = LabelEncoder()
        self.especie_encoder = LabelEncoder()

        #onde gêneros e especies acessam seus dados, pegando de posições do sample
        #posição 0 era a imagem
        generos = [s[1] for s in self.samples]
        especies = [s[2] for s in self.samples]

        self.genero_encoder.fit(generos)
        self.especie_encoder.fit(especies)

        # Mapeamento espécie → gênero (em índices numéricos)
        # um dicionário que mapeia 2 ids
        self.especie_to_genero = {
            #função sempre retorna vetor, o primeiro elemento é o id esperado
            self.especie_encoder.transform([esp])[0]: self.genero_encoder.transform([gen])[0]
            for _, gen, esp in self.samples
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # pegar dados da amostra segundo seu id
        path, genero, especie = self.samples[idx]
        img = Image.open(path).convert("RGB")
        # se tem transform, aplica (to tensor, por exemplo)
        if self.transform:
            img = self.transform(img)
        #obter id de gênero e espécie/ rótulos
        genero_label = self.genero_encoder.transform([genero])[0]
        especie_label = self.especie_encoder.transform([especie])[0]
        #tudo deve ser tensor...
        return img, torch.tensor(genero_label), torch.tensor(especie_label)
    
#funcao auxiliar para gerar dataloaders
def subset_from_paths(dataset, selected_paths):
    #paths é lista de strings indicando caminhos para imagens
    #por no subset as imagens presentes no dataset apontadas por ele
    #gera subsets seguros de serem acessados, função pronta não era reconhecido como dataset
    indices = [i for i, (p, _, _) in enumerate(dataset.samples) if p in selected_paths]
    #passa indices do dataset original
    return Subset(dataset, indices)


# =====================================================
# 2. Modelo com duas cabeças
# =====================================================

class SnakeClassifier(nn.Module):
    def __init__(self, backbone_name, n_generos, n_especies):
        super().__init__()
        #a linha abaixo pega o modelo que se fara o finetunning sobre
        #ambos os modelos surgiram do Imagenet
        base_model = getattr(models, backbone_name)(weights="IMAGENET1K_V1")
        #dados de entrada para a camada fc, de classificação, dependente do modelo ele é guardado em locais diferentes
        #Apesar de chamar o "classifier" aqui, é apenas para pegar o numero de entradas dele
        #esse número deverá 
        self.backbone_name = backbone_name
        #tanto resnet quanto resnext (o mesmo para efficientnet e efficientnet_v2)
        #segue a mesma estrutura de seu predecessor
        # efficient net é sequential e guarda in features de saída na estrutura "classifier"
        #segunda posição dele tem os dados desejados
        
        if "efficientnet" in self.backbone_name or "efficientnet_v2" in self.backbone_name:
            in_features = base_model.classifier[1].in_features 
        else:
            in_features = base_model.fc.in_features


        #remover a cabeça do antigo modelo (efficientnet / resnet) para a nossa classificação
        if hasattr(base_model, 'fc'):
            #do resnet
            base_model.fc = nn.Identity()
        elif hasattr(base_model, 'classifier'):
            #do efficientnet
            base_model.classifier = nn.Identity()
        #modelo já pré-treinado que se fara o finetunning, salva no modelo criado
        self.backbone = base_model
        #camadas fully connected separadas, cada uma gera saídas distintas, as nossas saídas 
        
        # Um pequeno bloco convolucional de refinamento
        if "efficientnet" in self.backbone_name or "efficientnet_v2" in self.backbone_name:
            self.refine = nn.Sequential( #é possível pegar dados do efficientnet sem flatten
                nn.Conv2d(in_channels=in_features, out_channels=512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Dropout2d(0.3),
                nn.AdaptiveAvgPool2d(1)  # reduz para (batch, 512, 1, 1)
            ) 
        else: #resnet aplica flatten antes de se obter os dados, diferente do efficientnet
            self.refine = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # saida classificadora
        self.fc_genero = nn.Linear(512, n_generos)
        self.fc_especie = nn.Linear(512, n_especies)

    
    def forward(self, x):
        if "efficientnet" in self.backbone_name or "efficientnet_v2" in self.backbone_name:
            x = self.backbone.features(x)  # em vez de self.backbone(x), o efficientnet
            x = self.refine(x) #precisa entrar no "features"
            x = torch.flatten(x, 1)
        else:
            x = self.backbone(x) #resnet já vem flat
            x = self.refine(x) #por isso não faz flatten aqui
        genero_logits = self.fc_genero(x)
        especie_logits = self.fc_especie(x)
        return genero_logits, especie_logits
    



# =====================================================
# 3. Funções de treino e avaliação
# =====================================================

def train_epoch(model, loader, optimizer, criterion, device, alfa = 0.3, beta = 0.7, scaler=None):
    model.train()
    total_loss = 0.0

    # Se o scaler não foi criado fora, cria aqui (melhor criar fora e reaproveitar entre epochs)
    # O scaler controla o gradiente para ele não estourar durante as iterações
    if scaler is None:
        scaler = amp.GradScaler('cuda')
    # percorrer loader de treino
    for imgs, y_gen, y_esp in tqdm(loader, desc="Treinando", leave=False):
        imgs, y_gen, y_esp = imgs.to(device), y_gen.to(device), y_esp.to(device)
        optimizer.zero_grad()

        # 1. Forward e cálculo ponderado da loss em função de alfa e beta
        with amp.autocast("cuda"):
            gen_logits, esp_logits = model(imgs)
            # calcula-se a loss de genero e especie separado
            loss_gen = criterion(gen_logits.float(), y_gen)
            loss_esp = criterion(esp_logits.float(), y_esp)
            # une-as em uma só loss ponderada
            loss = alfa* loss_gen + beta* loss_esp    

        # 2. Backward com GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader), scaler


def evaluate(model, loader, criterion, device, alfa = 0.3, beta = 0.7):
    model.eval()
    correct_gen, correct_esp, total = 0, 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for imgs, y_gen, y_esp in loader:
            imgs, y_gen, y_esp = imgs.to(device), y_gen.to(device), y_esp.to(device)
            # passa imagem, recebe os logits de genero e espécie
            gen_logits, esp_logits = model(imgs)
            #pega o maior do batch
            gen_pred = gen_logits.argmax(1)
            esp_pred = esp_logits.argmax(1)
            
            # Loss ponderada, como no treino
            loss_ext = criterion(gen_logits, y_gen.long())
            loss_int = criterion(esp_logits, y_esp.long())
            loss = alfa * loss_ext + beta * loss_int
            total_loss += loss.item()
            #contagem de acertos para calculo da acurácia
            correct_gen += (gen_pred == y_gen).sum().item()
            correct_esp += (esp_pred == y_esp).sum().item()

            total += imgs.size(0)

    avg_loss = total_loss / len(loader)
    #calculo da acurácia é semelhante a loss
    #no sentido de ser 2 unidas em uma ponderada
    acc_gen = correct_gen / total
    acc_esp = correct_esp / total
    acc_hier = (acc_gen * alfa + acc_esp * beta)
    return avg_loss, acc_gen, acc_esp, acc_hier


# =====================================================
# 4. Pipeline principal
# =====================================================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = "/mnt/c/Users/Enzo/Documents/Faculdade/RNA/DATASET_REDUZIDO_BALANCEADO"
    
    #obter dataset completo
    full_dataset = SnakeDataset(path, transform=transforms.ToTensor())

    #obter samples e separar seus dados
    samples = full_dataset.samples
    paths = [s[0] for s in samples]
    generos = [s[1] for s in samples]
    especies = [s[2] for s in samples]
    #gera labels hierarquicos, unido genero e espécie
    #zip percorre gêneros e espécies em paralelo, para fazer obter dados da label
    #pela forma que nosso código está estruturado, ficaria algo como Boa_Boa_constrictor
    #a repetição de gênero nesse nome não afeta nada, é só label
    labels_hier = [f"{g}_{e}" for g, e in zip(generos, especies)]

    #estratifica segundo label hierarquica
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths, especies, test_size=0.2, random_state=42, stratify=labels_hier
    )

    #temp paths é equivalente ao val e test unidos, aqui eles se separam
    val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    

    #particionar o dataset de forma segura, usar o random split dava conflito
    #random split não funciona com hierarquico ou estratificação
    train_dataset = subset_from_paths(full_dataset, train_paths)
    val_dataset   = subset_from_paths(full_dataset, val_paths)
    test_dataset  = subset_from_paths(full_dataset, test_paths)

    #o dataset de treino final possuí como dados a validação anterior, 
    #a nova "validação" é na verdade o teste
    train_final_dataset = ConcatDataset([train_dataset, val_dataset])

    #criação de loaders, apenas o treino tem shuffle
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=16) #
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=16)
    train_final_loader = DataLoader(train_final_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=16)
   
    writer = SummaryWriter(log_dir="runs/testeHML")   

    #usaremos a cross entropy loss    
    criterion = nn.CrossEntropyLoss()

    #loop para testes de diferentes Learning Rates
    np.random.seed(42)
    lr_ammount = 5
    lr_values = np.random.uniform(low=1e-5, high=5e-4, size=(lr_ammount))
    max_acc = 0.0 #escolheremos o lr do modelo de melhor acurácia
    best_lr = 0.0
    #best_lr = 0.00019352465823520762
    
    for i in range(lr_ammount):
        model = SnakeClassifier(
            backbone_name="efficientnet_v2_l", #"resnext101_32x8d" max = "resnext101_64x4d", "efficientnet_v2_l"
            n_generos=len(full_dataset.genero_encoder.classes_),
            n_especies=len(full_dataset.especie_encoder.classes_)
        ).to(device)
        
        #with open('modeloResNext.txt', 'w') as file:
        #    print(model, file=file)
        
        print(f" \n============ modelo {i+1} ============ ")
        
        #congelar todas as camadas, e depois descongela as últimas
        for param in model.backbone.parameters():
            param.requires_grad = False
    
        # descongelamento da última camada
        # A camada que deve ser descongelada é diferente entre a resNe(x)t e a EfficientNet(v2)
        # no resnet, a layer 3 tem mais convoluções que todas as outras somadas
        # em nossos testes, descongelar ela não foi eficiente
        if "resnext" in model.backbone_name or "resnet" in model.backbone_name:
            for name, param in model.backbone.named_parameters():
                if "layer4" in name:
                    param.requires_grad = True
        # no efficientnet tem vários blocos sequentials
        #novamente, descongelar só o último já foi suficiente
        elif "efficientnet_v2" in model.backbone_name or "efficientnet" in model.backbone_name:
            for name, param in model.backbone.named_parameters():
                if  "features.7" in name:
                    param.requires_grad = True

        # variáveis para coletar parâmetros do modelo
        backbone_params = []
        top_params = []

        #passar parametros do modelo e seu backbone para essas variáveis
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "backbone" in name:
                    backbone_params.append(param)
                else:
                    top_params.append(param)
        # passar os parâmetros otimizador Adam
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': 1e-5, 'weight_decay': 0.0},  # backbone com LR menor
            {'params': top_params, 'lr': lr_values[i], 'weight_decay': 1e-3},        # camadas novas
        ])

        criterion = nn.CrossEntropyLoss()
    
        scaler = amp.GradScaler('cuda')
        
        # Caso o modelo fique 3 épocas seguidas sem melhorar a loss,
        # é feita uma parada antecipada
        best_val_loss = float('inf') #guarda a melhor loss adquirida pelo modelo até então
        patience = 3     # número máximo de épocas sem melhora
        patience_counter = 0 # número de épocas sem melhora

        for epoch in range(10):
            # passagem pelo treino
            loss, scaler = train_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
            # validação
            val_loss, acc_gen, acc_esp, acc_hier = evaluate(model, val_loader, criterion , device)
            # visualização do dados
            print("\nepoch: ", epoch+1)
            print("loss: ", loss)
            print("val_loss: ", val_loss)
            print("acurácia gênero: ", acc_gen)
            print("acurácia espécie: ", acc_esp)
            print("acurácia hierarquica: ", acc_hier)
            
            # Escrita dos valores adquiridos durante a run para 
            # verificação dos gráficos usando TensorBoard
        
            #Escrita da loss no Writter
            loss_aux = "Train/LossModelo" + str(i)
            writer.add_scalars(loss_aux, {
                "Treino": loss,
                "Validação": val_loss
            }, epoch)
            
            #Escrita da acurácia no Writter
            acc_aux = "Train/AccuraciaModelo" + str(i)
            writer.add_scalars(acc_aux, {
                "Externa": acc_gen,
                "Interna": acc_esp,
                "Média": acc_hier
            }, epoch)
            
            # checagem de paciencia, modelo tem que melhorar a loss se ela piorar
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # salva o melhor modelo até agora
                torch.save(model.state_dict(), f"best_model_lr{i}.pth")
            else:
                patience_counter += 1
                print(f"→ Early stop patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(" Parando treino antecipadamente (early stopping ativado).")
                    break
          
        # Nota: O salvamento do modelo é feito de acordo com a loss, mas a escolha de LR é feita
        #       de acordo com a acurácia  
        
        #achar melhor modelo segundo acurácia (para learning rate)    
        if acc_hier > max_acc:
            max_acc = acc_hier
            best_lr = lr_values[i]
    
    print(f"Melhor LR: ", best_lr)
    
    print("-----TESTE FINAL-----")
    # repete
    model = SnakeClassifier(
            backbone_name="efficientnet_v2_l", #"resnext101_32x8d" max = "resnext101_64x4d", "efficientnet_v2_l"
            n_generos=len(full_dataset.genero_encoder.classes_),
            n_especies=len(full_dataset.especie_encoder.classes_)
    ).to(device)

    #congelar todas as camadas, e depois descongela as últimas
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # descongelamento da última camada
    if "resnext" in model.backbone_name or "resnet" in model.backbone_name:
        for name, param in model.backbone.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
    elif "efficientnet_v2" in model.backbone_name or "efficientnet" in model.backbone_name:
        for name, param in model.backbone.named_parameters():
            if "features.7" in name:
                param.requires_grad = True

    backbone_params = []
    top_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "backbone" in name:
                backbone_params.append(param)
            else:
                top_params.append(param)
    #usando otimizador Adam
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 1e-5, 'weight_decay': 0.0},  # backbone com LR menor
        {'params': top_params, 'lr': best_lr, 'weight_decay': 1e-3},        # camadas novas
    ])

    criterion = nn.CrossEntropyLoss()
    
    # estabilizador de gradientes
    scaler = amp.GradScaler('cuda')
    
    

    for epoch in range(25):
        loss, scaler = train_epoch(model, train_final_loader, optimizer, criterion, device, scaler=scaler)
        val_loss, acc_gen, acc_esp, acc_hier = evaluate(model, test_loader, criterion , device)
        print("epoch: ", epoch+1)
        print("loss: ", loss)
        print("acurácia gênero: ", acc_gen)
        print("acurácia espécie: ", acc_esp)
        print("acurácia hierarquica: ", acc_hier)
        
        loss_aux = "Final/LossModelo"
        writer.add_scalars(loss_aux, {
            "Treino_Final": loss,
            "Teste": val_loss
        }, epoch)
            
        acc_aux = "Final/AccuraciaModelo"
        writer.add_scalars(acc_aux, {
            "Externa": acc_gen,
            "Interna": acc_esp,
            "Média": acc_hier
        }, epoch)

    #salvar o modelo final
    torch.save(model.state_dict(), "snake_classifier_hier.pth")    
    
    
    #Gerar classification report
    model.eval()
    #gurada classes verdadeiras de genero e especie
    all_y_gen, all_y_esp = [], []
    #guarda a predição de generos e espécie
    all_pred_gen, all_pred_esp = [], []

    #realiza pequena passagem teste para obter variáveis
    with torch.no_grad():
        for imgs, y_gen, y_esp in test_loader:
            imgs = imgs.to(device)
            #obter logits do modelo (dados de teste)
            gen_logits, esp_logits = model(imgs)
            gen_pred = gen_logits.argmax(1).cpu()
            esp_pred = esp_logits.argmax(1).cpu()

            #adicionar todos os elementos de iterável (loader) 
            #feito de uma vez só com o extend
            #lembre! como temos batch, y_gen é um conjunto!
            all_y_gen.extend(y_gen.numpy())
            all_y_esp.extend(y_esp.numpy())
            all_pred_gen.extend(gen_pred.numpy())
            all_pred_esp.extend(esp_pred.numpy())

    # === GERA E SALVA AS MATRIZES ===
    #gera resumos (já impressos na função) e para serem passados na matriz confusão
    resumo_gen = resumo_f1(all_y_gen, all_pred_gen, nome="GÊNERO")
    resumo_esp = resumo_f1(all_y_esp, all_pred_esp, nome="ESPÉCIE")

    salvar_matriz_confusao(all_y_gen, all_pred_gen, full_dataset.genero_encoder, resumo_gen, tipo="GENERO")
    salvar_matriz_confusao(all_y_esp, all_pred_esp, full_dataset.especie_encoder, resumo_esp, tipo="ESPECIE")

    writer.close()


if __name__ == "__main__":
    main()
