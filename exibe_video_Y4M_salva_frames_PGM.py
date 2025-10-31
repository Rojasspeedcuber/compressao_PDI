import cv2
import os

# Caminho do video .y4m
video_path = "video/akiyo_cif.y4m"

# Nome da pasta de saida
output_folder = "frames_pgm"

# Cria a pasta se nao existir
os.makedirs(output_folder, exist_ok=True)

# Cria o objeto de captura
cap = cv2.VideoCapture(video_path)

# Verifica se o arquivo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o arquivo .y4m!")
    exit()

# Informacoes basicas do video
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"Resolucao: {width}x{height}")
print(f"Total de frames: {total_frames}")
print(f"Salvando frames em: {output_folder}/")

frame_count = 0

# Loop de leitura e salvamento
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do video ou erro de leitura.")
        break

    # Converte para escala de cinza (necessario para PGM)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gera o nome do arquivo com numeracao automatica
    filename = f"frame_{frame_count:03d}.pgm"
    filepath = os.path.join(output_folder, filename)

    # Salva o frame em formato PGM
    cv2.imwrite(filepath, gray_frame)

    # Exibe o frame atual
    cv2.imshow("Exibicao do video (.y4m)", frame)

    frame_count += 1

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()

print(f"\nProcesso concluido: {frame_count} frames salvos em '{output_folder}'")
