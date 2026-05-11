"""
=======================================================================
 ENTREGÁVEL 2 — ÁLGEBRA LINEAR
 Transformações Lineares sobre Coordenadas de Pixels
=======================================================================
 Transformações implementadas
 ─────────────────────────────
   1. Escalonamento (Scaling)       det(A) ≠ 0  →  isomorfismo
   2. Cisalhamento (Shear)          det(A) = 1  →  preserva área
   3. Rotação (Rotation)            det(A) = 1  →  isometria
   4. Colapso Dimensional           det(A) = 0  →  núcleo não-trivial
      (Projeção Ortogonal em eixo)

=============================================
"""


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ─────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO GERAL
# ─────────────────────────────────────────────────────────────────────

INPUT_IMAGE  = "arroz-branco-camil-t2-1kg-300x300.jpg"   # imagem original
OUTPUT_DIR   = "resultados"                               # pasta de saída
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# BLOCO 1 — DEFINIÇÃO DAS MATRIZES DE TRANSFORMAÇÃO
# ═════════════════════════════════════════════════════════════════════


def matriz_escalonamento(sx: float, sy: float) -> np.ndarray:

    A = np.array([[sx, 0 ],
                  [0,  sy]], dtype=float)
    print(f"[Escalonamento] A =\n{A}\n  det(A) = {np.linalg.det(A):.4f}")
    return A


def matriz_cisalhamento(kx: float = 0.0, ky: float = 0.0) -> np.ndarray:

    A = np.array([[1,  kx],
                  [ky,  1]], dtype=float)
    print(f"[Cisalhamento] A =\n{A}\n  det(A) = {np.linalg.det(A):.4f}")
    return A


def matriz_rotacao(angulo_graus: float) -> np.ndarray:

    theta = np.radians(angulo_graus)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = np.array([[ cos_t, -sin_t],
                  [ sin_t,  cos_t]], dtype=float)
    print(f"[Rotação {angulo_graus}°] A =\n{A}\n  det(A) = {np.linalg.det(A):.4f}")
    return A


def matriz_colapso_dimensional(eixo: str = 'x') -> np.ndarray:

    if eixo == 'x':
        A = np.array([[1, 0],
                      [0, 0]], dtype=float)
    else:  # eixo == 'y'
        A = np.array([[0, 0],
                      [0, 1]], dtype=float)
    print(f"[Colapso — projeção em '{eixo}'] A =\n{A}")
    print(f"  det(A) = {np.linalg.det(A):.4f}  ←  SINGULAR / Núcleo não-trivial")
    return A


# ═════════════════════════════════════════════════════════════════════
# BLOCO 2 — MOTOR DE TRANSFORMAÇÃO (Mapeamento Inverso)
# ═════════════════════════════════════════════════════════════════════


def aplicar_transformacao(img_array: np.ndarray,
                          A: np.ndarray,
                          metodo: str = "inverso",
                          bg_color: tuple = (255, 255, 255)
                          ) -> np.ndarray:

    H, W = img_array.shape[:2]

    cx, cy = W / 2, H / 2

    saida = np.full_like(img_array, bg_color, dtype=np.uint8)

    if metodo == "inverso":
        # ── Mapeamento Inverso ──────────────────────────────────────
        det = np.linalg.det(A)
        if abs(det) < 1e-10:
            raise ValueError(
                "det(A) ≈ 0: matriz singular. Use metodo='direto' "
                "para o caso de Colapso Dimensional."
            )
        A_inv = np.linalg.inv(A)

        # Gera malha de coordenadas de SAÍDA
        cols_out, rows_out = np.meshgrid(np.arange(W), np.arange(H))
        # Centraliza
        xp = cols_out.ravel() - cx
        yp = rows_out.ravel() - cy

        # Aplica A⁻¹ sobre cada ponto de saída
        coords_out = np.stack([xp, yp], axis=0)   # shape (2, H*W)
        coords_in  = A_inv @ coords_out           # shape (2, H*W)

        x_in = coords_in[0] + cx
        y_in = coords_in[1] + cy

        # Filtra coordenadas válidas (dentro da imagem de entrada)
        mask = (
            (x_in >= 0) & (x_in < W - 1) &
            (y_in >= 0) & (y_in < H - 1)
        )
        xi = np.round(x_in[mask]).astype(int)
        yi = np.round(y_in[mask]).astype(int)

        x_dst = cols_out.ravel()[mask]
        y_dst = rows_out.ravel()[mask]

        saida[y_dst, x_dst] = img_array[yi, xi]

    else:      # ── Mapeamento Direto (para matrizes singulares) ─────────────  
        rows_in, cols_in = np.arange(H), np.arange(W)
        C, R = np.meshgrid(cols_in, rows_in)

        xp = C.ravel() - cx
        yp = R.ravel() - cy

        coords_in  = np.stack([xp, yp], axis=0)
        coords_out = A @ coords_in

        x_out = np.round(coords_out[0] + cx).astype(int)
        y_out = np.round(coords_out[1] + cy).astype(int)

        mask = (
            (x_out >= 0) & (x_out < W) &
            (y_out >= 0) & (y_out < H)
        )
        saida[y_out[mask], x_out[mask]] = img_array[R.ravel()[mask], C.ravel()[mask]]

    return saida


# ═════════════════════════════════════════════════════════════════════
# BLOCO 3 — COMPARAÇÃO VISUAL (Matplotlib)
# ═════════════════════════════════════════════════════════════════════

def plotar_comparacao(img_original: np.ndarray,
                      transformadas: list,
                      titulos: list,
                      nome_arquivo: str,
                      det_values: list = None):
  
    n = len(transformadas)
    fig = plt.figure(figsize=(5 * (n + 1), 5))
    gs  = gridspec.GridSpec(1, n + 1, figure=fig)

    # ── Imagem original ──────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img_original)
    ax0.set_title("Original\n(Base Canônica intacta)", fontsize=11, fontweight='bold')
    ax0.axis('off')

    # ── Imagens transformadas ────────────────────────────────────────
    for i, (img_t, titulo) in enumerate(zip(transformadas, titulos)):
        ax = fig.add_subplot(gs[i + 1])
        ax.imshow(img_t)
        det_txt = ""
        if det_values is not None:
            det_txt = f"\ndet(A) = {det_values[i]:.4f}"
        ax.set_title(titulo + det_txt, fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.suptitle("Transformações Lineares sobre Coordenadas de Pixels",
                 fontsize=13, y=1.02, fontweight='bold')
    plt.tight_layout()

    caminho = os.path.join(OUTPUT_DIR, nome_arquivo)
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Figura salva: {caminho}")


# ═════════════════════════════════════════════════════════════════════
# BLOCO 4 — EXECUÇÃO PRINCIPAL
# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  ENTREGÁVEL 2 — Transformações Lineares")
    print("=" * 60)

    # ── Carrega imagem ───────────────────────────────────────────────
    img_pil   = Image.open(INPUT_IMAGE).convert("RGB")
    img_array = np.array(img_pil)
    H, W      = img_array.shape[:2]
    print(f"\nImagem carregada: {W}×{H} pixels\n")

    resultados   = []
    titulos      = []
    det_values   = []

    # ────────────────────────────────────────────────────────────────
    # TRANSFORMAÇÃO 1 — Escalonamento 0.6× horizontal / 1.3× vertical
    # ────────────────────────────────────────────────────────────────
    print("── Escalonamento ──")
    sx, sy = 0.6, 1.3
    A_esc  = matriz_escalonamento(sx, sy)

    img_esc = aplicar_transformacao(img_array, A_esc)
    Image.fromarray(img_esc).save(os.path.join(OUTPUT_DIR, "escalonamento.jpg"))
    resultados.append(img_esc)
    titulos.append(f"Escalonamento\n(sx={sx}, sy={sy})")
    det_values.append(np.linalg.det(A_esc))
    print()

    # ────────────────────────────────────────────────────────────────
    # TRANSFORMAÇÃO 2 — Cisalhamento horizontal 
    # ────────────────────────────────────────────────────────────────
    print("── Cisalhamento ──")
    kx    = 0.4
    A_cis = matriz_cisalhamento(kx=kx, ky=0.0)

    img_cis = aplicar_transformacao(img_array, A_cis)
    Image.fromarray(img_cis).save(os.path.join(OUTPUT_DIR, "cisalhamento.jpg"))
    resultados.append(img_cis)
    titulos.append(f"Cisalhamento\n(kx={kx})")
    det_values.append(np.linalg.det(A_cis))
    print()

    # ────────────────────────────────────────────────────────────────
    # TRANSFORMAÇÃO 3 — Rotação 45°
    # ────────────────────────────────────────────────────────────────
    print("── Rotação 45° ──")
    A_rot = matriz_rotacao(45)

    img_rot = aplicar_transformacao(img_array, A_rot)
    Image.fromarray(img_rot).save(os.path.join(OUTPUT_DIR, "rotacao_45.jpg"))
    resultados.append(img_rot)
    titulos.append("Rotação\n(θ = 45°)")
    det_values.append(np.linalg.det(A_rot))
    print()

    # ────────────────────────────────────────────────────────────────
    # TRANSFORMAÇÃO 4 — Colapso Dimensional (det = 0)
    # ────────────────────────────────────────────────────────────────
    print("── Colapso Dimensional (Projeção Ortogonal) ──")
    A_col = matriz_colapso_dimensional(eixo='x')

    # Mapeamento por método direto
    img_col = aplicar_transformacao(img_array, A_col, metodo="direto")
    Image.fromarray(img_col).save(os.path.join(OUTPUT_DIR, "colapso_dimensional.jpg"))
    resultados.append(img_col)
    titulos.append("Colapso Dimensional\n(Projeção eixo X, det=0)")
    det_values.append(np.linalg.det(A_col))
    print()

    # ────────────────────────────────────────────────────────────────
    # TRANSFORMAÇÃO 5 — Rotação 30°
    # ────────────────────────────────────────────────────────────────
    print("── Rotação 30° ──")
    A_rot30 = matriz_rotacao(30)
    img_rot30 = aplicar_transformacao(img_array, A_rot30)
    Image.fromarray(img_rot30).save(os.path.join(OUTPUT_DIR, "rotacao_30.jpg"))
    print()

    # ── Figura comparativa ───────────────────────────────────────────
    print("── Gerando figura comparativa ──")
    plotar_comparacao(
        img_array,
        resultados,
        titulos,
        nome_arquivo="comparacao_transformacoes.png",
        det_values=det_values
    )

    # ── Figura exclusiva: Colapso ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_array)
    axes[0].set_title("Original — 2D (rank = 2)", fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(img_col)
    axes[1].set_title(
        "Após Projeção Ortogonal\ndet(A) = 0  |  rank(A) = 1  |  nullity = 1",
        fontweight='bold', color='darkred'
    )
    axes[1].axis('off')
    plt.suptitle("Colapso Dimensional: perda irreversível de informação", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "colapso_dimensional_destaque.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Figura salva: {os.path.join(OUTPUT_DIR, 'colapso_dimensional_destaque.png')}")

    # ── Resumo numérico ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESUMO DAS TRANSFORMAÇÕES")
    print("=" * 60)
    nomes = ["Escalonamento (sx=0.6, sy=1.3)",
             "Cisalhamento  (kx=0.4)",
             "Rotação       (θ=45°)",
             "Colapso       (proj. X)"]
    matrizes = [A_esc, A_cis, A_rot, A_col]
    for nome, A, d in zip(nomes, matrizes, det_values):
        invertivel = "Invertível ✓" if abs(d) > 1e-10 else "SINGULAR ✗ (Colapso)"
        print(f"\n  {nome}")
        print(f"    Matriz A  = {A.tolist()}")
        print(f"    det(A)    = {d:.6f}")
        print(f"    Status    = {invertivel}")

    print("\n✅ Todos os arquivos salvos em:", OUTPUT_DIR)


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
