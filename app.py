```python
import time

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
import joblib
import pandas as pd
from skimage.feature import hog


st.set_page_config(
    page_title="Detecção de Placas - YOLOv11 x HOG + SVM",
    layout="wide"
)


@st.cache_resource
def carregar_yolo():
    return YOLO("models/best.pt")


@st.cache_resource
def carregar_svm():
    return joblib.load("models/svm_model.pkl")


def carregar_imagem(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        # Reduz o tamanho para deixar o HOG + SVM mais viável no Streamlit
        image.thumbnail((960, 960))

        return np.array(image)

    except Exception as e:
        st.error("Não foi possível carregar essa imagem.")
        st.exception(e)
        st.stop()


def extrair_hog(crop_rgb):
    crop_resized = cv2.resize(crop_rgb, (128, 64))
    gray = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


def desenhar_box(img_rgb, box, texto, cor=(0, 255, 0)):
    img_box = img_rgb.copy()

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        img_box,
        (x1, y1),
        (x2, y2),
        cor,
        3
    )

    cv2.putText(
        img_box,
        texto,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        cor,
        2
    )

    return img_box


def detectar_yolo(img_rgb, model_yolo, conf_threshold):
    inicio = time.time()

    results = model_yolo.predict(
        img_rgb,
        conf=conf_threshold,
        verbose=False
    )

    fim = time.time()
    tempo = fim - inicio

    deteccoes = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_rgb.shape[1], x2)
            y2 = min(img_rgb.shape[0], y2)

            deteccoes.append({
                "box": (x1, y1, x2, y2),
                "conf": conf
            })

    if len(deteccoes) == 0:
        return None, 0.0, tempo, 0

    melhor = max(deteccoes, key=lambda d: d["conf"])

    return melhor["box"], melhor["conf"], tempo, len(deteccoes)


def detectar_hog_svm(
    img_rgb,
    svm_model,
    window_size=(128, 64),
    stride=40,
    scales=None
):
    if scales is None:
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    inicio = time.time()

    h_img, w_img = img_rgb.shape[:2]
    base_w, base_h = window_size

    features_list = []
    boxes = []

    for scale in scales:
        win_w = int(base_w * scale)
        win_h = int(base_h * scale)

        if win_w >= w_img or win_h >= h_img:
            continue

        for y in range(0, h_img - win_h, stride):
            for x in range(0, w_img - win_w, stride):
                crop = img_rgb[y:y + win_h, x:x + win_w]

                if crop.size == 0:
                    continue

                features = extrair_hog(crop)

                features_list.append(features)
                boxes.append((x, y, x + win_w, y + win_h))

    total_janelas = len(boxes)

    if total_janelas == 0:
        fim = time.time()
        return None, None, total_janelas, fim - inicio

    X = np.array(features_list)

    preds = svm_model.predict(X)

    if hasattr(svm_model, "decision_function"):
        scores = svm_model.decision_function(X)
    else:
        scores = preds.astype(float)

    indices_placa = np.where(preds == 1)[0]

    if len(indices_placa) == 0:
        fim = time.time()
        maior_score = float(np.max(scores))
        return None, maior_score, total_janelas, fim - inicio

    melhor_indice = indices_placa[np.argmax(scores[indices_placa])]

    melhor_box = boxes[melhor_indice]
    melhor_score = float(scores[melhor_indice])

    fim = time.time()
    tempo = fim - inicio

    return melhor_box, melhor_score, total_janelas, tempo


st.title("Detecção de Placas Veiculares")
st.markdown("### Comparação entre YOLOv11 e HOG + SVM")

st.write(
    """
    Envie uma imagem de veículo para comparar dois métodos de detecção de placas:
    **YOLOv11**, baseado em Deep Learning, e **HOG + SVM**, baseado em Machine Learning clássico
    com varredura por janelas deslizantes.
    """
)


with st.sidebar:
    st.header("Sobre o Projeto")
    st.write("Disciplina: Machine Learning")
    st.write("Modelo 1: YOLOv11")
    st.write("Modelo 2: HOG + SVM")
    st.write("Classe detectada: placa")

    conf_threshold = st.slider(
        "Confiança mínima do YOLO",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    st.divider()

    st.subheader("Configuração HOG + SVM")

    stride_hog = st.select_slider(
        "Passo da janela",
        options=[24, 32, 40, 48, 56],
        value=40
    )

    st.caption(
        "Quanto menor o passo, mais janelas são avaliadas. Isso pode melhorar a busca, mas deixa o processamento mais lento."
    )


try:
    yolo_model = carregar_yolo()
    svm_model = carregar_svm()
except Exception as e:
    st.error("Erro ao carregar os modelos.")
    st.exception(e)
    st.stop()


uploaded_file = st.file_uploader(
    "Envie uma imagem",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is None:
    st.info("Envie uma imagem para iniciar a detecção.")

else:
    img_np = carregar_imagem(uploaded_file)

    st.subheader("Imagem enviada")
    st.image(img_np, use_container_width=True)

    with st.spinner("Executando YOLOv11..."):
        box_yolo, conf_yolo, tempo_yolo, total_yolo = detectar_yolo(
            img_np,
            yolo_model,
            conf_threshold
        )

    with st.spinner("Executando HOG + SVM com Sliding Window..."):
        box_hog, score_hog, total_janelas_hog, tempo_hog = detectar_hog_svm(
            img_np,
            svm_model,
            window_size=(128, 64),
            stride=stride_hog,
            scales=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        )

    st.subheader("Resultado da Detecção")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### YOLOv11")

        if box_yolo is not None:
            img_yolo = desenhar_box(
                img_np,
                box_yolo,
                f"YOLO placa {conf_yolo:.2f}",
                cor=(0, 255, 0)
            )

            st.image(
                img_yolo,
                caption="Detecção do YOLOv11 na imagem completa",
                use_container_width=True
            )

            st.success("Placa detectada pelo YOLOv11.")

            st.metric(
                label="Confiança YOLO",
                value=f"{conf_yolo * 100:.2f}%"
            )

            st.write(f"**Tempo de inferência:** {tempo_yolo:.2f} s")
            st.write(f"**Total de detecções:** {total_yolo}")

            x1, y1, x2, y2 = box_yolo

            st.write(
                f"""
                **Bounding box:**  
                x1={x1}, y1={y1}  
                x2={x2}, y2={y2}
                """
            )

        else:
            st.image(
                img_np,
                caption="Nenhuma placa detectada pelo YOLOv11",
                use_container_width=True
            )

            st.warning("Nenhuma placa foi detectada pelo YOLOv11.")
            st.write(f"**Tempo de inferência:** {tempo_yolo:.2f} s")

    with col2:
        st.markdown("### HOG + SVM")

        if box_hog is not None:
            img_hog = desenhar_box(
                img_np,
                box_hog,
                f"HOG+SVM {score_hog:.2f}",
                cor=(255, 0, 0)
            )

            st.image(
                img_hog,
                caption="Detecção do HOG + SVM por Sliding Window",
                use_container_width=True
            )

            st.success("Placa detectada pelo HOG + SVM.")

            st.metric(
                label="Score SVM",
                value=f"{score_hog:.4f}"
            )

            st.write(f"**Tempo de varredura:** {tempo_hog:.2f} s")
            st.write(f"**Janelas avaliadas:** {total_janelas_hog}")

            x1, y1, x2, y2 = box_hog

            st.write(
                f"""
                **Bounding box:**  
                x1={x1}, y1={y1}  
                x2={x2}, y2={y2}
                """
            )

        else:
            st.image(
                img_np,
                caption="Nenhuma placa detectada pelo HOG + SVM",
                use_container_width=True
            )

            st.warning("Nenhuma placa foi detectada pelo HOG + SVM.")

            if score_hog is not None:
                st.metric(
                    label="Maior score encontrado",
                    value=f"{score_hog:.4f}"
                )

            st.write(f"**Tempo de varredura:** {tempo_hog:.2f} s")
            st.write(f"**Janelas avaliadas:** {total_janelas_hog}")

    st.subheader("Comparação dos Modelos")

    resultado_yolo = (
        f"Detectou placa com {conf_yolo * 100:.2f}% de confiança"
        if box_yolo is not None
        else "Não detectou placa"
    )

    resultado_hog = (
        f"Detectou placa com score {score_hog:.4f}"
        if box_hog is not None
        else "Não detectou placa"
    )

    tabela_comparacao = pd.DataFrame({
        "Modelo": ["YOLOv11", "HOG + SVM"],
        "Abordagem": [
            "Deep Learning - detecção de objetos",
            "Machine Learning clássico - Sliding Window"
        ],
        "Entrada": [
            "Imagem completa",
            "Imagem completa"
        ],
        "Função": [
            "Localizar a placa diretamente",
            "Varrer a imagem com janelas e classificar cada região"
        ],
        "Resultado na imagem enviada": [
            resultado_yolo,
            resultado_hog
        ],
        "Tempo": [
            f"{tempo_yolo:.2f} s",
            f"{tempo_hog:.2f} s"
        ]
    })

    st.table(tabela_comparacao)

    st.info(
        """
        O YOLOv11 é um detector de objetos treinado para localizar a placa diretamente.
        O HOG + SVM foi treinado como classificador de placa/não placa e aplicado como detector
        por meio de janelas deslizantes. Por isso, tende a ser mais lento e mais sensível ao tamanho
        da janela e à escala da placa.
        """
    )
```
