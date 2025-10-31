import cv2
import os
import natsort  # para ordenar os nomes de forma natural (frame_1, frame_2, ...)

# Caminho da pasta com os frames .pgm
input_folder = "frames_pgm"

# Verifica se a pasta existe
if not os.path.exists(input_folder):
    print(f"Erro: pasta '{input_folder}' não encontrada!")
    exit()

# Lista todos os arquivos .pgm na pasta
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pgm")]

# Ordena os arquivos pelo nome (ordem natural)
image_files = natsort.natsorted(image_files)

# Verifica se há imagens
if len(image_files) == 0:
    print("Nenhuma imagem .pgm encontrada na pasta!")
    exit()

print(f"{len(image_files)} imagens encontradas em '{input_folder}'.")

# Loop para exibir as imagens em sequência
for filename in image_files:
    # Caminho completo do arquivo
    filepath = os.path.join(input_folder, filename)

    # Lê a imagem
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    # Verifica se a imagem foi carregada corretamente
    if img is None:
        print(f"[ERRO] Falha ao abrir {filepath}")
        continue

    # Exibe a imagem
    cv2.imshow("Sequência de imagens (.pgm)", img)
    cv2.setWindowTitle("Sequência de imagens (.pgm)", f"Exibindo: {filename}")

    # Espera 30 ms entre cada imagem (~33 fps)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Encerra a exibição
cv2.destroyAllWindows()
print("Exibição concluída.")
