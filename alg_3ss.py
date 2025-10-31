"""
Implementação 3SS block-matching + reconstrução IPIP... + PSNR
Autor: (exemplo)
Requisitos: numpy, opencv-python
"""
import os
import math
import cv2
import numpy as np
from typing import List, Tuple, Dict

# ---------------------------
# Utilitários de I/O e Y (luminância)
# ---------------------------
def load_frames_from_folder(folder: str, ext_list: List[str] = None) -> List[np.ndarray]:
    """
    Carrega todas as imagens da pasta, ordenadas alfabeticamente.
    Retorna lista de frames (uint8) contendo a componente de luminância (Y) com shape (H, W).
    """
    if ext_list is None:
        ext_list = ['.png', '.jpg', '.jpeg', '.bmp', '.pgm']
    files = [f for f in sorted(os.listdir(folder)) if os.path.splitext(f)[1].lower() in ext_list]
    frames = []
    for f in files:
        path = os.path.join(folder, f)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # Se multi-canal, converter para Y (luminância)
        if len(img.shape) == 3 and img.shape[2] == 3:
            # OpenCV lê BGR -> converter para YCrCb e pegar Y
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y = ycrcb[:, :, 0]
        else:
            # Assumir já grayscale
            y = img
        # Garantir tipo uint8
        if y.dtype != np.uint8:
            y = y.astype(np.uint8)
        frames.append(y)
    return frames

# ---------------------------
# PSNR
# ---------------------------
def psnr_frame(original: np.ndarray, reconstructed: np.ndarray, max_pixel: float = 255.0) -> float:
    """
    Calcula PSNR entre duas imagens uint8 (luminância).
    Retorna valor em dB. Se MSE == 0 retorna float('inf')
    """
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10((max_pixel ** 2) / mse)

# ---------------------------
# Helpers para blocos e padding
# ---------------------------
def pad_frame(frame: np.ndarray, block_size: int) -> Tuple[np.ndarray, int, int]:
    """
    Faz padding (replicação nas bordas) para que altura e largura sejam múltiplos de block_size.
    Retorna (frame_padded, pad_h, pad_w)
    pad_h é o número de linhas adicionadas, pad_w é colunas adicionadas.
    """
    h, w = frame.shape
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    if pad_h == 0 and pad_w == 0:
        return frame, 0, 0
    frame_padded = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    return frame_padded, pad_h, pad_w

# ---------------------------
# Three-Step Search (3SS)
# ---------------------------
def initial_step_for_max_disp(max_disp: int) -> int:
    """
    Calcula o passo inicial (potência de dois) apropriado para max_disp.
    Escolha: step_init = 2**(ceil(log2(max_disp+1)) - 1)
    Ex.: max_disp=7 -> ceil(log2(8))=3 -> 2**(3-1)=4 -> passos: 4,2,1
          max_disp=15->ceil(log2(16))=4->2**3=8 -> passos: 8,4,2 (mas depois força ao menos 1 em último passo)
    """
    if max_disp <= 0:
        return 1
    n = math.ceil(math.log2(max_disp + 1))
    step = 2 ** max(0, n - 1)
    return max(1, step)

def three_step_search_block(ref: np.ndarray, cur: np.ndarray, top: int, left: int,
                            block_size: int, max_disp: int) -> Tuple[int,int]:
    """
    Executa 3SS para um bloco localizado em (top,left) na frame 'cur', buscando em 'ref'
    Retorna o vetor de movimento (dy, dx) que minimiza SAD, com restrições de max_disp.
    ref e cur são arrays 2D (uint8). top/left indicam posição do bloco na cur (assumimos já padronizados).
    """
    h, w = cur.shape
    blk_h = block_size
    blk_w = block_size

    # Bloco atual
    cur_blk = cur[top:top+blk_h, left:left+blk_w].astype(np.int32)

    step = initial_step_for_max_disp(max_disp)
    best_mv = (0, 0)
    # centro inicial é mv = (0,0)
    # calcular custo inicial (no deslocamento 0)
    best_cost = np.sum(np.abs(cur_blk - ref[top:top+blk_h, left:left+blk_w].astype(np.int32)))

    # Limites possíveis para deslocamento (em relação ao top,left)
    # max_disp é deslocamento máximo absoluto (tanto + quanto -)
    # durante busca, quando avaliarmos posições fora dos limites da imagem, vamos
    # clipar a posição de amostragem em ref (ou alternativamente ignorar essas posições).
    # Aqui usaremos clip para evitar exceções e garantir pesquisa.
    for _ in range(3):  # exatamente três passos
        found_better = False
        # varrer -step, 0, +step
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                cand_dy = best_mv[0] + dy
                cand_dx = best_mv[1] + dx
                # respeitar max_disp
                if abs(cand_dy) > max_disp or abs(cand_dx) > max_disp:
                    continue
                # posição top-left no referencial da ref
                ref_top = top + cand_dy
                ref_left = left + cand_dx
                # clip para borda (para evitar indexing fora)
                ref_top_clipped = min(max(ref_top, 0), h - blk_h)
                ref_left_clipped = min(max(ref_left, 0), w - blk_w)
                ref_blk = ref[ref_top_clipped:ref_top_clipped+blk_h, ref_left_clipped:ref_left_clipped+blk_w].astype(np.int32)
                cost = np.sum(np.abs(cur_blk - ref_blk))
                if cost < best_cost:
                    best_cost = cost
                    best_mv = (cand_dy, cand_dx)
                    found_better = True
        # reduzir passo (dividir por 2, mantendo pelo menos 1)
        step = max(1, step // 2)
        # continue para próximo passo independentemente de ter encontrado melhor ou não,
        # já que 3SS tradicional faz exatamente 3 passos.
    return best_mv

# ---------------------------
# Motion estimation para um frame inteiro por blocos
# ---------------------------
def estimate_motion_3ss(ref: np.ndarray, cur: np.ndarray, block_size: int, max_disp: int) -> np.ndarray:
    """
    Estima vetores de movimento por blocos usando 3SS.
    Retorna um array de shape (num_blocks_y, num_blocks_x, 2) de vetores (dy,dx).
    Ambos 'ref' e 'cur' devem ter dimensões múltiplas de block_size (use pad_frame antes).
    """
    h, w = cur.shape
    nby = h // block_size
    nbx = w // block_size
    mvs = np.zeros((nby, nbx, 2), dtype=np.int32)
    for by in range(nby):
        top = by * block_size
        for bx in range(nbx):
            left = bx * block_size
            dy, dx = three_step_search_block(ref, cur, top, left, block_size, max_disp)
            mvs[by, bx, 0] = dy
            mvs[by, bx, 1] = dx
    return mvs

# ---------------------------
# Motion compensation (reconstruir frame P a partir do ref usando MVs)
# ---------------------------
def motion_compensate(ref: np.ndarray, mvs: np.ndarray, block_size: int) -> np.ndarray:
    """
    Gera frame estimado (reconstruído) aplicando os vetores de movimento mvs sobre frame 'ref'.
    ref shape (H,W); mvs shape (nby, nbx, 2)
    """
    h, w = ref.shape
    nby, nbx, _ = mvs.shape
    recon = np.zeros_like(ref)
    for by in range(nby):
        top = by * block_size
        for bx in range(nbx):
            left = bx * block_size
            dy, dx = int(mvs[by, bx, 0]), int(mvs[by, bx, 1])
            ref_top = top + dy
            ref_left = left + dx
            # clip
            ref_top_clipped = min(max(ref_top, 0), h - block_size)
            ref_left_clipped = min(max(ref_left, 0), w - block_size)
            block = ref[ref_top_clipped:ref_top_clipped+block_size, ref_left_clipped:ref_left_clipped+block_size]
            recon[top:top+block_size, left:left+block_size] = block
    return recon

# ---------------------------
# Reconstruir sequência com estrutura IPIP...
# ---------------------------
def reconstruct_sequence_IP(frames: List[np.ndarray], block_size: int, max_disp: int) -> Tuple[List[np.ndarray], Dict]:
    """
    frames: lista de frames originais (luminância) como uint8 arrays.
    block_size: 8 ou 16 (ou outro)
    max_disp: 7,15,24,31
    Retorna (reconstructed_frames, stats) onde reconstructed_frames é lista de frames reconstruídos (uint8).
    stats contém PSNR por frame e média, e número de frames I/P.
    Estrutura: frames[0] será I (copiado), frames[1] -> P (estimado a partir de recon[0]), frames[2] -> I (copiado), etc.
    """
    # Preparar frames (padding) todos para mesmo tamanho múltiplo de block_size
    padded_frames = []
    pad_h_total = pad_w_total = 0
    for f in frames:
        fpad, ph, pw = pad_frame(f, block_size)
        padded_frames.append(fpad)
        pad_h_total = max(pad_h_total, ph)
        pad_w_total = max(pad_w_total, pw)
    # Garantir que todos os frames tenham o mesmo padding aplicado (replicar novamente se necessário)
    final_padded = []
    for f in padded_frames:
        ph = pad_h_total - (f.shape[0] - frames[0].shape[0]) if pad_h_total > 0 else 0
        pw = pad_w_total - (f.shape[1] - frames[0].shape[1]) if pad_w_total > 0 else 0
        if ph > 0 or pw > 0:
            f2 = cv2.copyMakeBorder(f, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
        else:
            f2 = f
        final_padded.append(f2)
    # Reconstruir
    reconstructed = []
    psnr_list = []
    stats = {"I_frames": 0, "P_frames": 0}
    for idx, orig in enumerate(final_padded):
        if idx % 2 == 0:
            # I-frame: copiamos original (intra-coded)
            rec = orig.copy()
            reconstructed.append(rec)
            stats["I_frames"] += 1
        else:
            # P-frame: estimar movimento a partir do último frame reconstruído
            ref = reconstructed[-1]
            mvs = estimate_motion_3ss(ref, orig, block_size, max_disp)
            rec = motion_compensate(ref, mvs, block_size)
            reconstructed.append(rec)
            stats["P_frames"] += 1
        # calcular PSNR entre 'orig' (sem padding extra sobre o original real) e 'rec'
        # Remover eventuais padding adicionados (recorte para tamanho original)
        # tamanho original do frame antes de padding:
        h0, w0 = frames[idx].shape
        rec_crop = reconstructed[-1][:h0, :w0]
        orig_crop = frames[idx][:h0, :w0]
        psnr_val = psnr_frame(orig_crop, rec_crop)
        psnr_list.append(psnr_val)
    stats["psnr_per_frame"] = psnr_list
    stats["psnr_mean"] = float(np.mean([p for p in psnr_list if not math.isinf(p)])) if len(psnr_list) > 0 else float('nan')
    return reconstructed, stats

# ---------------------------
# Função principal de simulação (para dois blocosizes e vários deslocamentos)
# ---------------------------
def simulate_for_sequence(folder: str, block_sizes: List[int], max_disps: List[int]) -> Dict:
    """
    Executa toda a simulação para a sequência na pasta 'folder' para as combinações de block_size e max_disp.
    Retorna dicionário com resultados.
    """
    frames = load_frames_from_folder(folder)
    if len(frames) == 0:
        raise ValueError(f"Nenhum frame encontrado em {folder}")
    results = {}
    for bs in block_sizes:
        for md in max_disps:
            key = f"bs{bs}_md{md}"
            print(f"Simulando {folder} -> bloco {bs} max_disp {md} ...")
            recon, stats = reconstruct_sequence_IP(frames, block_size=bs, max_disp=md)
            results[key] = {"reconstructed_frames": recon, "stats": stats}
    return results


if __name__ == "__main__":
    import pprint

    # Caminho da pasta contendo os frames ordenados (ex: frame0001.png, frame0002.png, ...)
    # ➤ Altere para o caminho da sua pasta:
    folder = "frames_pgm"  # exemplo: "./Akiyo" ou "./Stefan"

    # Tamanhos de bloco e deslocamentos máximos a testar
    block_sizes = [8, 16]
    max_disps = [7, 15, 24, 31]

    print(f"\n=== Iniciando simulação 3SS para pasta: {folder} ===\n")

    # Executa a simulação (gera reconstrução + PSNR)
    results = simulate_for_sequence(folder, block_sizes, max_disps)

    # Exibe o resumo dos resultados (PSNR médio por configuração)
    print("\n=== Resultados de PSNR médio por configuração ===\n")
    for key, data in results.items():
        psnr_mean = data["stats"]["psnr_mean"]
        nI = data["stats"]["I_frames"]
        nP = data["stats"]["P_frames"]
        print(f"{key}: PSNR médio = {psnr_mean:.2f} dB  | I-frames: {nI} | P-frames: {nP}")

    # Exemplo: acessar frames reconstruídos e salvar um deles (opcional)
    # Aqui pegamos o primeiro frame reconstruído da configuração bs8_md7
    recon_frames = results["bs8_md7"]["reconstructed_frames"]
    primeiro_frame = recon_frames[0]
    cv2.imwrite("frame_reconstruido_exemplo.png", primeiro_frame)
    print("\nFrame reconstruído de exemplo salvo como 'frame_reconstruido_exemplo.png'.\n")

    # Se quiser imprimir detalhes completos (PSNR de cada frame):
    # pprint.pp(results["bs8_md7"]["stats"]["psnr_per_frame"])
