import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import joblib
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


def extrair_hog(crop):
    crop = cv2.resize(crop, (128, 64))
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.reshape(1, -1)


def desenhar_box(img, x1, y1, x2, y2, conf):
    img_box = img.copy()

    cv2.rectangle(img_box, (x1, y1), (x2, y2), (0, 255, 0), 3)

    texto = f"placa {conf:.2f}"
    cv2.putText(
        img_box,
        texto,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    return img_box


st.title("Detecção de Placas Veiculares")
st.markdown("### Comparação entre YOLOv11 e HOG + SVM")

st.write(
    """
    Envie uma imagem de veículo para que o sistema localize a placa com YOLOv11,
    recorte a região detectada e valide o recorte utilizando HOG + SVM.
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


try:
    yolo_model = carregar_yolo()
    svm_model = carregar_svm()
except Exception as e:
    st.error("Erro ao carregar os modelos.")
    st.exception(e)
    st.stop()


uploaded_file = st.file_uploader(
    "Envie uma imagem",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is None:
    st.info("Envie uma imagem para iniciar a detecção.")

else:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.subheader("Imagem enviada")
    st.image(img_np, use_container_width=True)

    with st.spinner("Executando YOLOv11..."):
        results = yolo_model.predict(
            img_np,
            conf=conf_threshold,
            verbose=False
        )

    deteccoes = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_yolo = float(box.conf[0])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_np.shape[1], x2)
            y2 = min(img_np.shape[0], y2)

            crop = img_np[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            features = extrair_hog(crop)
            pred_svm = svm_model.predict(features)[0]

            if hasattr(svm_model, "decision_function"):
                score_svm = float(svm_model.decision_function(features)[0])
            else:
                score_svm = None

            label_svm = "placa" if pred_svm == 1 else "não placa"

            deteccoes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf_yolo": conf_yolo,
                "crop": crop,
                "label_svm": label_svm,
                "score_svm": score_svm
            })

    if len(deteccoes) == 0:
        st.warning("Nenhuma placa foi detectada pelo YOLOv11.")

    else:
        det_principal = max(deteccoes, key=lambda d: d["conf_yolo"])

        img_com_box = desenhar_box(
            img_np,
            det_principal["x1"],
            det_principal["y1"],
            det_principal["x2"],
            det_principal["y2"],
            det_principal["conf_yolo"]
        )

        st.subheader("Resultado da Detecção")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### YOLOv11")
            st.image(img_com_box, caption="Imagem com bounding box", use_container_width=True)

            st.metric(
                label="Confiança YOLO",
                value=f"{det_principal['conf_yolo'] * 100:.2f}%"
            )

            st.write(
                f"""
                **Bounding box:**  
                x1={det_principal['x1']}, y1={det_principal['y1']},  
                x2={det_principal['x2']}, y2={det_principal['y2']}
                """
            )

        with col2:
            st.markdown("### HOG + SVM")
            st.image(
                det_principal["crop"],
                caption="Recorte analisado pelo SVM",
                use_container_width=False
            )

            if det_principal["label_svm"] == "placa":
                st.success("Classificação: placa")
            else:
                st.error("Classificação: não placa")

            if det_principal["score_svm"] is not None:
                st.metric(
                    label="Score SVM",
                    value=f"{det_principal['score_svm']:.4f}"
                )

        st.subheader("Comparação dos Modelos")

        st.table({
            "Modelo": ["YOLOv11", "HOG + SVM"],
            "Entrada": ["Imagem completa", "Recorte detectado pelo YOLO"],
            "Função": ["Localizar a placa", "Classificar o recorte"],
            "Resultado": [
                f"Placa detectada com {det_principal['conf_yolo'] * 100:.2f}% de confiança",
                f"Classificado como {det_principal['label_svm']}"
            ]
        })

        if len(deteccoes) > 1:
            st.info(f"Foram encontradas {len(deteccoes)} detecções. Foi exibida a de maior confiança.")
