import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
try:
    from langchain_text_splitters import CharacterTextSplitter
except ImportError:
    from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except ImportError:
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.chat_models import ChatOpenAI
import platform

# ---- Estilo personalizado: paleta azul petróleo + terracota (Miguel Ruiz) ----
try:
    _icon = Image.open('logo_mr.png')
except Exception:
    _icon = "📄"
st.set_page_config(page_title="Chat con PDF · Miguel Ruiz", page_icon=_icon, layout="centered")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700&family=Inter:wght@300;400;600&display=swap');

    /* Fondo azul petróleo oscuro */
    .stApp {
        background: linear-gradient(135deg, #1b3a4b 0%, #14303d 100%);
        color: #e8eef0;
        font-family: 'Inter', sans-serif;
    }
    /* Texto general claro: párrafos, listas, spans, markdown */
    .stApp, .stApp p, .stApp li, .stApp span, .stApp label,
    .stMarkdown, [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {
        color: #e8eef0 !important;
    }
    /* Títulos en terracota */
    h1, h2, h3, h4 {
        font-family: 'Montserrat', sans-serif !important;
        color: #e07a5f !important;
    }
    /* Campos de entrada: fondo claro con texto oscuro para que se lean */
    .stTextInput input, .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input {
        background-color: #f3f6f7 !important;
        color: #14303d !important;
    }
    input::placeholder, textarea::placeholder { color: #5c7079 !important; }

    /* Mensajes de estado (info/warning/success) legibles */
    [data-testid="stNotification"] * { color: #14303d !important; }

    /* Barra lateral azul petróleo más oscuro */
    section[data-testid="stSidebar"] {
        background: #12272f;
    }
    /* Todo el texto de la barra lateral en claro */
    section[data-testid="stSidebar"] * {
        color: #e8eef0 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e07a5f !important;
    }
    /* Inputs dentro de la barra lateral: fondo claro, texto oscuro */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #f3f6f7 !important;
        color: #14303d !important;
    }
    /* Botones terracota */
    .stButton>button {
        background-color: #e07a5f;
        color: #fff !important;
        border-radius: 10px;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #c9583c;
        color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and presentation
st.markdown("<h1 style=\"font-family:'Montserrat', sans-serif; font-weight:700; font-size:2.4rem; color:#f2cc8f; margin-bottom:0;\">Miguel Ruiz</h1>", unsafe_allow_html=True)
st.title('Generación Aumentada por Recuperación (RAG) 💬')
st.caption("Conversa con tus documentos PDF usando Inteligencia Artificial.")
st.write("Versión de Python:", platform.python_version())

# Load and display image
try:
    image = Image.open('chat_pdf_img.jpg')
    st.image(image, width=350, caption="Tus documentos, a un chat de distancia 📚")
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar information
with st.sidebar:
    st.subheader("📄 Agente lector de PDF")
    st.write("Este agente te ayudará a realizar análisis y responder preguntas sobre el PDF que cargues. Desarrollado por Miguel Ruiz.")

# Get API key from user
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# PDF uploader
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Texto extraído: {len(text)} caracteres")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")
        
        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # User question interface
        st.subheader("✨ Escribe qué quieres saber sobre el documento")
        user_question = st.text_area("Tu pregunta", placeholder="Escribe tu pregunta aquí...")
        
        # Process question when submitted
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Modelo de chat actual (compatible con gpt-4o-mini)
            llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
            
            # Load QA chain
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Run the chain
            response = chain.run(input_documents=docs, question=user_question)
            
            # Display the response
            st.markdown("### Respuesta:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        # Add detailed error for debugging
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
