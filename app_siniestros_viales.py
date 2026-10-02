from pathlib import Path
import html
import pickle

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Envigado Vial",
    page_icon="🚦",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #f3f6fa;
    color: #10263d;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, p, label {
    font-family: "Segoe UI", sans-serif;
}

.marca {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 5px 24px;
}

.logo {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 27px;
    font-weight: 750;
    color: #10263d;
}

.semaforo {
    background: #10263d;
    border-radius: 8px;
    width: 27px;
    padding: 5px 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
}

.semaforo i {
    height: 9px;
    width: 9px;
    border-radius: 50%;
    display: block;
}

.academico {
    background: #e3edf6;
    padding: 9px 18px;
    border-radius: 25px;
    font-size: 14px;
    color: #10263d;
}

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(115deg, #10263d, #14384c);
    border-radius: 20px;
    padding: 38px 40px;
    margin-bottom: 24px;
}

.hero .eyebrow {
    color: #f6cc55;
    letter-spacing: 3px;
    font-size: 13px;
    font-weight: 750;
}

.hero h1 {
    color: white;
    font-size: clamp(32px, 4vw, 54px);
    line-height: 1.12;
    margin: 14px 0 9px;
    padding: 0;
    position: relative;
    z-index: 1;
}

.hero h2 {
    color: white;
    font-size: clamp(19px, 2.2vw, 27px);
    margin: 0 0 20px;
    padding: 0;
    position: relative;
    z-index: 1;
}

.hero p {
    color: #c7d7e2;
    margin: 0;
    max-width: 780px;
    font-size: 17px;
    position: relative;
    z-index: 1;
}

.carretera {
    position: absolute;
    right: -20px;
    top: -50px;
    width: 260px;
    height: 330px;
    opacity: .6;
}

div[data-testid="stForm"],
.st-key-resultado {
    background: white;
    border: 1px solid #e8edf3;
    border-radius: 20px;
    padding: 26px;
    box-shadow: 0 8px 30px rgba(16, 38, 61, .025);
}

.st-key-resultado {
    min-height: 740px;
}

.titulo-form {
    font-size: 25px;
    font-weight: 750;
    color: #10263d;
    margin-bottom: 5px;
}

.subtitulo {
    color: #66778e;
    font-size: 15px;
    margin-bottom: 22px;
}

.seccion {
    display: flex;
    gap: 12px;
    align-items: center;
    color: #10263d;
    font-weight: 700;
    font-size: 18px;
    margin: 14px 0 15px;
}

.numero {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: #087f80;
    color: white;
    font-size: 14px;
    flex-shrink: 0;
}

div[data-testid="stWidgetLabel"] p {
    color: #10263d !important;
    font-size: 14px;
}

div[data-baseweb="select"] > div {
    background: #fbfcfe;
    border-color: #d7e0ea;
    border-radius: 9px;
    min-height: 44px;
    color: #10263d;
}

div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(100deg, #087f74, #0c8990);
    color: white;
    border: none;
    border-radius: 11px;
    min-height: 50px;
    font-size: 17px;
    font-weight: 700;
    margin-top: 13px;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: #06665e;
    color: white;
}

.pill {
    display: inline-block;
    background: #fff0c9;
    color: #735013;
    padding: 8px 15px;
    border-radius: 24px;
    font-size: 12px;
    font-weight: 650;
    letter-spacing: .7px;
}

.resultado-titulo {
    font-size: 26px;
    color: #10263d;
    font-weight: 750;
    text-align: center;
    margin: 25px 0;
}

.insignia {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    margin: 30px auto;
    display: flex;
    align-items: center;
    justify-content: center;
}

.gravedad {
    font-size: clamp(25px, 3vw, 35px);
    font-weight: 800;
    text-align: center;
    margin: 25px 0 15px;
}

.descripcion {
    color: #586b80;
    line-height: 1.6;
    font-size: 17px;
    text-align: center;
    margin-bottom: 30px;
}

.nota {
    border-top: 1px solid #e0e6ee;
    padding: 23px 0;
    color: #64748b;
    font-size: 15px;
    line-height: 1.6;
}

.franja {
    background: #e7eff7;
    border-radius: 14px;
    padding: 20px 25px;
    margin-top: 22px;
    color: #516680;
    font-size: 16px;
}

.pie {
    border-top: 1px solid #dce5ef;
    text-align: center;
    padding-top: 20px;
    margin-top: 24px;
    color: #7b8ba0;
    font-size: 13px;
}

@media(max-width: 700px) {
    .hero {padding: 26px 22px;}
    .carretera {opacity: .2;}
    .academico {font-size: 11px; padding: 8px;}
    .logo {font-size: 21px;}
    .st-key-resultado {min-height: 0;}
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="marca">
    <div class="logo">
        <span class="semaforo">
            <i style="background:#f05252"></i>
            <i style="background:#f6cc55"></i>
            <i style="background:#11b985"></i>
        </span>
        Envigado Vial
    </div>
    <span class="academico">Proyecto académico</span>
</div>

<div class="hero">
    <svg class="carretera" viewBox="0 0 260 330" aria-hidden="true">
        <path d="M180 -20 C300 110 -50 135 150 350"
              stroke="#278ea2" stroke-width="3" fill="none"/>
        <path d="M210 -20 C330 110 -20 135 180 350"
              stroke="#f6cc55" stroke-width="3"
              stroke-dasharray="13 11" fill="none"/>
        <path d="M240 -20 C360 110 10 135 210 350"
              stroke="#10b999" stroke-width="3" fill="none"/>
    </svg>
    <div class="eyebrow">SEGURIDAD VIAL</div>
    <h1>Cada dato cuenta.</h1>
    <h2>Estimación de la gravedad de siniestros viales</h2>
    <p>
        Completa las características del evento
        y consulta la clasificación estimada.
    </p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_modelo():
    ruta = Path(__file__).parent / "modelo-siniestros.pkl"

    with open(ruta, "rb") as archivo:
        modelo, encoder, variables, scaler, opciones = pickle.load(archivo)

    return modelo, encoder, list(variables), scaler, opciones


def preparar_entrada(entrada, variables, scaler):
    fila = pd.DataFrame(0.0, index=[0], columns=variables)

    numericas = ["HORA_DIA", "RESULTADO DE BEODEZ"]
    fila.loc[:, numericas] = scaler.transform(entrada[numericas])

    categoricas = [
        "DÍA DE LA SEMANA",
        "CLASE DE ACCIDENTE",
        "AREA",
        "MES",
        "CAUSA_AGRUPADA",
        "BARRIO_AGRUPADO"
    ]

    for columna in categoricas:
        dummy = f"{columna}_{entrada.iloc[0][columna]}"
        if dummy in fila.columns:
            fila.loc[0, dummy] = 1.0

    return fila


def seccion(numero, titulo):
    st.markdown(
        f"""
        <div class="seccion">
            <span class="numero">{numero}</span>
            {titulo}
        </div>
        """,
        unsafe_allow_html=True
    )


try:
    modelo, encoder, variables, scaler, opciones = cargar_modelo()

except FileNotFoundError:
    st.error(
        "No se encontró modelo-siniestros.pkl. "
        "Ubícalo en la misma carpeta de la aplicación."
    )
    st.stop()

except Exception as error:
    st.error(f"No se pudo cargar el modelo: {error}")
    st.stop()


izquierda, derecha = st.columns([1.65, 1], gap="medium")


with izquierda:
    with st.form("siniestro"):
        st.markdown("""
        <div class="titulo-form">Características del siniestro</div>
        <div class="subtitulo">Selecciona la información registrada.</div>
        """, unsafe_allow_html=True)

        seccion("01", "Cuándo ocurrió")

        a, b = st.columns(2)

        meses = [
            "Enero", "Febrero", "Marzo", "Abril",
            "Mayo", "Junio", "Julio", "Agosto",
            "Septiembre", "Octubre", "Noviembre", "Diciembre"
        ]

        with a:
            mes = st.selectbox(
                "Mes",
                opciones["MES"],
                format_func=lambda valor: meses[int(valor) - 1]
            )

            hora = st.selectbox(
                "Hora",
                range(24),
                index=12,
                format_func=lambda valor: f"{valor:02d}:00"
            )

        with b:
            dia = st.selectbox(
                "Día de la semana",
                opciones["DÍA DE LA SEMANA"]
            )

        st.divider()
        seccion("02", "Dónde ocurrió")

        a, b = st.columns(2)

        with a:
            area = st.selectbox("Área", opciones["AREA"])

        with b:
            barrio = st.selectbox(
                "Barrio",
                opciones["BARRIO_AGRUPADO"]
            )

        st.divider()
        seccion("03", "Características del evento")

        a, b = st.columns(2)

        with a:
            clase = st.selectbox(
                "Clase de accidente",
                opciones["CLASE DE ACCIDENTE"]
            )

            beodez = st.selectbox(
                "Resultado de beodez",
                [0, 1, 2, 3],
                format_func=lambda valor: (
                    "0 · Negativo"
                    if valor == 0
                    else f"{valor} · Grado {valor}"
                )
            )

        with b:
            causa = st.selectbox(
                "Causa registrada",
                opciones["CAUSA_AGRUPADA"]
            )

        consultar = st.form_submit_button(
            "Estimar gravedad  →",
            use_container_width=True
        )

    if consultar:
        entrada = pd.DataFrame([{
            "DÍA DE LA SEMANA": dia,
            "CLASE DE ACCIDENTE": clase,
            "AREA": area,
            "MES": mes,
            "HORA_DIA": hora,
            "RESULTADO DE BEODEZ": beodez,
            "CAUSA_AGRUPADA": causa,
            "BARRIO_AGRUPADO": barrio
        }])

        try:
            with st.spinner("Analizando el evento..."):
                entrada_modelo = preparar_entrada(
                    entrada, variables, scaler
                )

                prediccion = modelo.predict(entrada_modelo)[0]

                gravedad = str(
                    encoder.inverse_transform([int(prediccion)])[0]
                )

            st.session_state["resultado_vial"] = {
                "gravedad": gravedad,
                "entrada": entrada
            }

        except Exception as error:
            st.session_state.pop("resultado_vial", None)
            st.error(f"No se pudo realizar la estimación: {error}")


with derecha:
    with st.container(key="resultado", border=False):
        resultado = st.session_state.get("resultado_vial")

        pill = (
            "RESULTADO DE LA ESTIMACIÓN"
            if resultado
            else "LISTO PARA CONSULTAR"
        )

        st.markdown(
            f"""
            <div style="text-align:center">
                <span class="pill">{pill}</span>
            </div>
            <div class="resultado-titulo">Gravedad estimada</div>
            """,
            unsafe_allow_html=True
        )

        if resultado:
            gravedad = resultado["gravedad"]
            danos = gravedad == "SOLO DAÑOS"

            color, fondo = (
                ("#087f3d", "#def7e9")
                if danos
                else ("#b45309", "#fff0d6")
            )

            simbolo = "✓" if danos else "!"

            descripcion = (
                "El modelo estima un siniestro con daños materiales."
                if danos
                else "El modelo estima un siniestro con personas heridas."
            )

            escudo = f"""
            <svg width="85" height="95" viewBox="0 0 85 95"
                 aria-hidden="true">
                <path d="M42 3 L78 18 V49 Q78 75 42 91
                         Q6 75 6 49 V18 Z"
                      fill="{color}"/>
                <text x="42" y="64" text-anchor="middle"
                      fill="white" font-family="sans-serif"
                      font-size="48" font-weight="bold">
                    {simbolo}
                </text>
            </svg>
            """

            st.markdown(
                f"""
                <div class="insignia" style="background:{fondo}">
                    {escudo}
                </div>
                <div class="gravedad" style="color:{color}">
                    {html.escape(gravedad)}
                </div>
                <div class="descripcion">{descripcion}</div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.markdown("""
            <div class="insignia"
                 style="background:#e7eff7;color:#547089;font-size:60px">
                ⌕
            </div>
            <div class="descripcion">
                Completa el formulario y pulsa
                <b>Estimar gravedad</b> para consultar el resultado.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="nota">
            ⓘ &nbsp; La estimación no confirma
            las consecuencias reales del evento.
        </div>
        """, unsafe_allow_html=True)

        if resultado:
            with st.expander("Ver datos utilizados"):
                tabla = (
                    resultado["entrada"].T
                    .rename(columns={0: "Valor registrado"})
                    .astype(str)
                )

                st.dataframe(tabla, use_container_width=True)
                st.caption("Datos de la última estimación enviada.")


st.markdown("""
<div class="franja">
    ⓘ &nbsp; El modelo clasifica entre
    <b>SOLO DAÑOS</b> y <b>HERIDOS</b>.
</div>
<div class="pie">
    Minería de datos · Siniestros viales de Envigado
</div>
""", unsafe_allow_html=True)
